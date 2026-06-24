from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import ScriptTask
from .llm import LLM
from .surprise import JudgeScores, build_llm_judge_prompt, parse_llm_judge_response, rule_judge, should_stop


@dataclass
class PipelineConfig:
    max_branches: int = 2
    min_branches: int = 3
    max_active_branches: int = 2
    fork_enabled: bool = True
    fork_score_threshold: float = 3.85
    active_prune_margin: float = 0.75
    similarity_prune_threshold: float = 0.72
    max_scenes: int = 4
    min_scenes: int = 3
    max_new_tokens: int = 768
    scene_max_new_tokens: int = 768
    temperature: float = 0.7
    judge_backend: str = "llm"


@dataclass
class PipelineStats:
    api_calls: int = 0
    judge_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    branch_count: int = 0
    stopped_branch_count: int = 0
    stopped_scene_count: int = 0
    wall_time: float = 0.0
    generation_diagnostics: list[dict[str, float]] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __post_init__(self) -> None:
        self.generation_diagnostics = self.generation_diagnostics or []

    def add_generation(
        self,
        generation_input_tokens: int,
        generation_output_tokens: int,
        diagnostics: dict[str, float] | None = None,
    ) -> None:
        self.api_calls += 1
        self.input_tokens += generation_input_tokens
        self.output_tokens += generation_output_tokens
        if diagnostics:
            self.generation_diagnostics.append(diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "judge_calls": self.judge_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "branch_count": self.branch_count,
            "stopped_branch_count": self.stopped_branch_count,
            "stopped_scene_count": self.stopped_scene_count,
            "wall_time": round(self.wall_time, 3),
            "generation_diagnostics": _summarize_generation_diagnostics(self.generation_diagnostics or []),
        }


@dataclass
class PipelineResult:
    task_id: str
    final_script: str
    record: dict[str, Any]


def _summarize_generation_diagnostics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    summary = {}
    for key in keys:
        values = [float(item[key]) for item in items if key in item]
        if values:
            summary[f"{key}_mean"] = round(sum(values) / len(values), 4)
            summary[f"{key}_max"] = round(max(values), 4)
    summary["count"] = len(items)
    return summary


@dataclass
class BranchState:
    branch_id: int
    parent_id: int | None = None
    status: str = "active"
    stage: str = "seed"
    seed: str = ""
    seed_data: dict[str, Any] | None = None
    storyline: str = ""
    storyline_data: dict[str, Any] | None = None
    characters: str = ""
    characters_data: dict[str, Any] | None = None
    outline: str = ""
    outline_data: dict[str, Any] | None = None
    scene_goals: list[str] | None = None
    scenes: list[dict[str, Any]] | None = None
    scores: dict[str, dict[str, Any]] | None = None
    decisions: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.seed_data = self.seed_data or {}
        self.storyline_data = self.storyline_data or {}
        self.characters_data = self.characters_data or {}
        self.outline_data = self.outline_data or {}
        self.scene_goals = self.scene_goals or []
        self.scenes = self.scenes or []
        self.scores = self.scores or {}
        self.decisions = self.decisions or []

    def latest_score(self) -> dict[str, Any]:
        if not self.scores:
            return {}
        for key in ["final", "scene", "outline", "characters", "storyline", "seed"]:
            if key in self.scores:
                return self.scores[key]
        return {}

    def generated_text(self) -> str:
        return "\n\n".join(str(scene.get("text", "")) for scene in self.scenes or []).strip()


class ScriptPipeline:
    def __init__(self, llm: LLM, config: PipelineConfig) -> None:
        self.llm = llm
        self.config = config

    def run_task(self, task: ScriptTask) -> PipelineResult:
        started = time.time()
        stats = PipelineStats()

        def call(prompt: str, max_new_tokens: int | None = None, temperature: float | None = None) -> str:
            generation = self.llm.generate(
                prompt,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                temperature=self.config.temperature if temperature is None else temperature,
            )
            stats.add_generation(generation.input_tokens, generation.output_tokens, generation.diagnostics)
            return generation.text.strip()

        raw_outputs: dict[str, Any] = {}

        task_card_raw = call(_normalization_prompt(task), temperature=0.2)
        raw_outputs["task_card"] = task_card_raw
        task_card_data = _coerce_task_card(task, task_card_raw)
        task_card = _compact_json(task_card_data)

        controller_events: list[dict[str, Any]] = []
        branches: list[BranchState] = []
        previous_seed = ""
        initial_count = min(max(self.config.min_branches, 1), self.config.max_branches)
        for idx in range(1, initial_count + 1):
            seed_raw = call(_seed_prompt(task, task_card, idx, previous_seed))
            raw_outputs[f"branch_{idx}_seed"] = seed_raw
            seed_data = _coerce_seed(seed_raw, idx)
            seed = _format_seed(seed_data)
            score = self._judge(seed, task.user_prompt, previous_seed, stats)
            branch = BranchState(branch_id=idx, seed=seed, seed_data=seed_data)
            branch.scores["seed"] = score.to_dict()
            invalid, invalid_reason = _invalid_reason(score, stage="branch")
            branch.status = "pruned" if invalid else "active"
            branch.decisions.append(_decision_event("seed", "prune" if invalid else "keep", invalid_reason or "initial_branch", score.to_dict()))
            branches.append(branch)
            stats.branch_count += 1
            previous_seed += "\n" + seed
            controller_events.append(_controller_event("INIT_BRANCH", branch.branch_id, "seed", branch.status, score.to_dict(), invalid_reason or "initial_seed"))

        active = _controller_prune(
            branches,
            self.config.max_active_branches,
            controller_events,
            "seed",
            self.config.active_prune_margin,
            self.config.similarity_prune_threshold,
        )

        for branch in list(active):
            storyline_raw = call(_storyline_prompt(task, task_card, branch.seed))
            raw_outputs[f"branch_{branch.branch_id}_storyline"] = storyline_raw
            branch.storyline_data = _coerce_storyline(storyline_raw)
            branch.storyline = _format_storyline(branch.storyline_data)
            score = self._judge(branch.storyline, _branch_judge_context(task, branch, "storyline"), branch.seed, stats)
            branch.scores["storyline"] = score.to_dict()
            branch.stage = "storyline"
            branch.decisions.append(_decision_event("storyline", "keep", "expanded_storyline", score.to_dict()))
            controller_events.append(_controller_event("EXPAND_STORYLINE", branch.branch_id, "storyline", branch.status, score.to_dict(), "depth+1"))
        if self.config.fork_enabled:
            for parent in list(active):
                if stats.branch_count >= self.config.max_branches:
                    break
                if _branch_rank(parent) < self.config.fork_score_threshold:
                    continue
                if any(branch.parent_id == parent.branch_id for branch in branches):
                    continue
                new_id = max(branch.branch_id for branch in branches) + 1
                siblings = "\n".join(branch.seed for branch in branches)
                fork_raw = call(_fork_seed_prompt(task, task_card, parent, new_id, siblings), temperature=min(self.config.temperature + 0.15, 1.0))
                raw_outputs[f"branch_{new_id}_fork_seed"] = fork_raw
                fork_seed_data = _coerce_seed(fork_raw, new_id)
                fork_seed = _format_seed(fork_seed_data)
                seed_score = self._judge(fork_seed, task.user_prompt, siblings, stats)
                fork_branch = BranchState(branch_id=new_id, parent_id=parent.branch_id, seed=fork_seed, seed_data=fork_seed_data)
                fork_branch.scores["seed"] = seed_score.to_dict()
                fork_branch.decisions.append(_decision_event("seed", "fork", f"fork_from_branch_{parent.branch_id}", seed_score.to_dict()))
                branches.append(fork_branch)
                stats.branch_count += 1
                controller_events.append(_controller_event("FORK_BRANCH", fork_branch.branch_id, "storyline", "active", seed_score.to_dict(), f"parent={parent.branch_id}"))

                storyline_raw = call(_storyline_prompt(task, task_card, fork_branch.seed))
                raw_outputs[f"branch_{fork_branch.branch_id}_storyline"] = storyline_raw
                fork_branch.storyline_data = _coerce_storyline(storyline_raw)
                fork_branch.storyline = _format_storyline(fork_branch.storyline_data)
                storyline_score = self._judge(fork_branch.storyline, _branch_judge_context(task, fork_branch, "storyline"), fork_branch.seed, stats)
                fork_branch.scores["storyline"] = storyline_score.to_dict()
                fork_branch.stage = "storyline"
                fork_branch.decisions.append(_decision_event("storyline", "keep", "expanded_fork_storyline", storyline_score.to_dict()))
                controller_events.append(_controller_event("EXPAND_STORYLINE", fork_branch.branch_id, "storyline", fork_branch.status, storyline_score.to_dict(), "fork_depth+1"))

        active = _controller_prune(
            branches,
            self.config.max_active_branches,
            controller_events,
            "storyline",
            self.config.active_prune_margin,
            self.config.similarity_prune_threshold,
        )

        for branch in list(active):
            characters_raw = call(_character_prompt(task, task_card, branch.storyline))
            raw_outputs[f"branch_{branch.branch_id}_characters"] = characters_raw
            branch.characters_data = _coerce_characters(characters_raw)
            branch.characters = _format_characters(branch.characters_data)
            score = self._judge(branch.characters, _branch_judge_context(task, branch, "characters"), branch.storyline, stats)
            branch.scores["characters"] = score.to_dict()
            branch.stage = "characters"
            branch.decisions.append(_decision_event("characters", "keep", "expanded_characters", score.to_dict()))
            controller_events.append(_controller_event("EXPAND_CHARACTERS", branch.branch_id, "characters", branch.status, score.to_dict(), "depth+1"))
        active = _controller_prune(
            branches,
            self.config.max_active_branches,
            controller_events,
            "characters",
            self.config.active_prune_margin,
            self.config.similarity_prune_threshold,
        )

        for branch in list(active):
            outline_raw = call(_outline_prompt(task, task_card, branch.storyline, branch.characters))
            raw_outputs[f"branch_{branch.branch_id}_outline"] = outline_raw
            branch.outline_data = _coerce_outline(outline_raw)
            branch.outline = _format_outline(branch.outline_data)
            branch.scene_goals = _extract_scene_goals(branch.outline, self.config.max_scenes, branch.outline_data)
            score = self._judge(branch.outline, _branch_judge_context(task, branch, "outline"), branch.characters, stats)
            branch.scores["outline"] = score.to_dict()
            branch.stage = "outline"
            branch.decisions.append(_decision_event("outline", "keep", "expanded_outline", score.to_dict()))
            controller_events.append(_controller_event("EXPAND_OUTLINE", branch.branch_id, "outline", branch.status, score.to_dict(), "depth+1"))
        active = _controller_prune(
            branches,
            self.config.max_active_branches,
            controller_events,
            "outline",
            self.config.active_prune_margin,
            self.config.similarity_prune_threshold,
        )

        for scene_idx in range(1, self.config.max_scenes + 1):
            scene_active = [branch for branch in active if branch.status == "active"]
            if not scene_active:
                break
            for branch in scene_active:
                if scene_idx > len(branch.scene_goals):
                    branch.status = "completed"
                    branch.decisions.append(_decision_event(f"scene_{scene_idx}", "complete", "no_more_scene_goals", branch.latest_score()))
                    controller_events.append(_controller_event("COMPLETE_BRANCH", branch.branch_id, f"scene_{scene_idx}", branch.status, branch.latest_score(), "no_more_scene_goals"))
                    continue
                scene_goal = branch.scene_goals[scene_idx - 1]
                generated_so_far = branch.generated_text()
                scene_raw = call(
                    _scene_prompt(
                        task=task,
                        task_card=task_card,
                        storyline=branch.storyline,
                        characters=branch.characters,
                        outline=branch.outline,
                        previous_text=generated_so_far,
                        scene_idx=scene_idx,
                        scene_goal=scene_goal,
                        total_scene_goals=len(branch.scene_goals or []),
                    ),
                    max_new_tokens=self.config.scene_max_new_tokens,
                )
                raw_outputs[f"branch_{branch.branch_id}_scene_{scene_idx}"] = scene_raw
                scene_data = _coerce_scene(scene_raw, scene_idx, scene_goal, branch.characters_data)
                scene_text = _format_scene(scene_data)
                scene_judge_context = _scene_judge_context(task, scene_goal, branch.characters, branch.outline)
                score = self._judge(scene_text, scene_judge_context, generated_so_far, stats)
                score_dict = score.to_dict()
                branch.scores["scene"] = score_dict
                stop, reason = _branch_scene_stop(branch, score, scene_idx, self.config.min_scenes)
                decision = "accept_and_stop" if stop else "accept"
                branch.scenes.append(
                    {
                        "scene_id": scene_idx,
                        "goal": scene_goal,
                        "raw_text": scene_raw,
                        "scene_data": scene_data,
                        "text": scene_text,
                        "score": score_dict,
                        "decision": decision,
                        "decision_reason": reason,
                    }
                )
                branch.stage = f"scene_{scene_idx}"
                if stop:
                    branch.status = "completed"
                    stats.stopped_scene_count += 1
                branch.decisions.append(_decision_event(f"scene_{scene_idx}", decision, reason, score_dict))
                controller_events.append(_controller_event("EXPAND_SCENE", branch.branch_id, f"scene_{scene_idx}", branch.status, score_dict, reason))
            active = _controller_prune(
                branches,
                self.config.max_active_branches,
                controller_events,
                f"scene_{scene_idx}",
                self.config.active_prune_margin,
                self.config.similarity_prune_threshold,
            )

        for branch in branches:
            if branch.status == "active" and branch.scenes:
                branch.status = "completed"
                branch.decisions.append(_decision_event("max_depth", "complete", "reached_max_scenes", branch.latest_score()))
                controller_events.append(_controller_event("COMPLETE_BRANCH", branch.branch_id, "max_depth", branch.status, branch.latest_score(), "reached_max_scenes"))

        candidates = [branch for branch in branches if branch.scenes and branch.status != "pruned"]
        if not candidates:
            candidates = [branch for branch in branches if branch.scenes] or branches
        for branch in candidates:
            final_text = _compose_final_script(task, branch.characters, branch.outline, branch.scenes)
            final_score = self._final_judge(final_text, _final_judge_context(task, branch), stats)
            final_score = _calibrate_final_completion(final_score, branch)
            branch.scores["final"] = final_score.to_dict()
            branch.decisions.append(_decision_event("final", "candidate", "final_judge", final_score.to_dict()))
            controller_events.append(_controller_event("FINAL_JUDGE", branch.branch_id, "final", branch.status, final_score.to_dict(), "candidate"))

        best_final_branch = max(candidates, key=_branch_rank)
        final_script = _compose_final_script(task, best_final_branch.characters, best_final_branch.outline, best_final_branch.scenes)
        final_score = JudgeScores(
            novelty=float(best_final_branch.scores.get("final", {}).get("novelty", 3)),
            relevance=float(best_final_branch.scores.get("final", {}).get("relevance", 3)),
            plot_progress=float(best_final_branch.scores.get("final", {}).get("plot_progress", 3)),
            logic_consistency=float(best_final_branch.scores.get("final", {}).get("logic_consistency", 3)),
            character_consistency=float(best_final_branch.scores.get("final", {}).get("character_consistency", 3)),
            overall_quality=float(best_final_branch.scores.get("final", {}).get("overall_quality", 3)),
            comments=str(best_final_branch.scores.get("final", {}).get("comments", "selected_final")),
        )
        controller_events.append(_controller_event("SELECT_FINAL", best_final_branch.branch_id, "final", "selected", final_score.to_dict(), "best_rank"))
        stats.stopped_branch_count = len([branch for branch in branches if branch.status == "pruned"])
        stats.wall_time = time.time() - started

        record = {
            "task": {
                "id": task.id,
                "genre": task.genre,
                "theme": task.theme,
                "user_prompt": task.user_prompt,
                "constraints": task.constraints,
            },
            "task_card": task_card,
            "task_card_data": task_card_data,
            "branches": [_branch_to_record(branch) for branch in branches],
            "controller_events": controller_events,
            "selected_branch_id": best_final_branch.branch_id,
            "chosen_seed": best_final_branch.seed,
            "storyline": best_final_branch.storyline,
            "storyline_data": best_final_branch.storyline_data,
            "characters": best_final_branch.characters,
            "characters_data": best_final_branch.characters_data,
            "outline": best_final_branch.outline,
            "outline_data": best_final_branch.outline_data,
            "scenes": best_final_branch.scenes,
            "final_score": final_score.to_dict(),
            "metrics": stats.to_dict(),
            "raw_outputs": raw_outputs,
        }
        return PipelineResult(task_id=task.id, final_script=final_script, record=record)

    def _judge(self, candidate: str, task_prompt: str, previous_text: str, stats: PipelineStats) -> JudgeScores:
        if self.config.judge_backend == "llm":
            prompt = build_llm_judge_prompt(candidate, task_prompt, previous_text)
            generation = self.llm.generate(prompt, max_new_tokens=512, temperature=0.0)
            stats.add_generation(generation.input_tokens, generation.output_tokens, generation.diagnostics)
            stats.judge_calls += 1
            return parse_llm_judge_response(generation.text, candidate, task_prompt, previous_text)
        return rule_judge(candidate, task_prompt, previous_text)

    def _final_judge(self, candidate: str, task_prompt: str, stats: PipelineStats) -> JudgeScores:
        if self.config.judge_backend in {"llm", "hybrid"}:
            prompt = build_llm_judge_prompt(candidate, task_prompt, "")
            generation = self.llm.generate(prompt, max_new_tokens=512, temperature=0.0)
            stats.add_generation(generation.input_tokens, generation.output_tokens, generation.diagnostics)
            stats.judge_calls += 1
            return parse_llm_judge_response(generation.text, candidate, task_prompt, "")
        return rule_judge(candidate, task_prompt, "")


def _controller_prune(
    branches: list[BranchState],
    max_active: int,
    controller_events: list[dict[str, Any]],
    stage: str,
    active_prune_margin: float,
    similarity_prune_threshold: float,
) -> list[BranchState]:
    active = [branch for branch in branches if branch.status == "active"]
    active.sort(key=_branch_rank, reverse=True)
    if not active:
        return []
    best_rank = _branch_rank(active[0])
    weak_prune = [
        branch for branch in active[1:]
        if best_rank - _branch_rank(branch) > active_prune_margin and len(active) > 1
    ]
    for branch in weak_prune:
        branch.status = "pruned"
        score = branch.latest_score()
        branch.decisions.append(_decision_event(stage, "prune", "active_prune_margin", score))
        controller_events.append(_controller_event("PRUNE_BRANCH", branch.branch_id, stage, "pruned", score, "active_prune_margin"))

    active = [branch for branch in active if branch.status == "active"]
    active.sort(key=_branch_rank, reverse=True)
    for i, keeper in enumerate(active):
        for candidate in active[i + 1:]:
            if candidate.status != "active":
                continue
            similarity = _branch_similarity(keeper, candidate)
            if similarity >= similarity_prune_threshold:
                candidate.status = "pruned"
                score = candidate.latest_score()
                reason = f"similar_to_branch_{keeper.branch_id}:{similarity:.2f}"
                candidate.decisions.append(_decision_event(stage, "prune", reason, score))
                controller_events.append(_controller_event("PRUNE_BRANCH", candidate.branch_id, stage, "pruned", score, reason))

    active = [branch for branch in active if branch.status == "active"]
    active.sort(key=_branch_rank, reverse=True)
    keep = active[:max_active]
    prune = active[max_active:]
    for branch in prune:
        branch.status = "pruned"
        score = branch.latest_score()
        branch.decisions.append(_decision_event(stage, "prune", "controller_width_limit", score))
        controller_events.append(_controller_event("PRUNE_BRANCH", branch.branch_id, stage, "pruned", score, "controller_width_limit"))
    return keep


def _branch_rank(branch: BranchState) -> float:
    latest = branch.latest_score()
    final_bonus = 0.4 if "final" in (branch.scores or {}) else 0.0
    depth_bonus = min(len(branch.scenes or []), 4) * 0.08
    diversity_bonus = _branch_diversity_bonus(branch)
    gain_bonus = _branch_gain_bonus(branch)
    return _score_rank(latest) + depth_bonus + diversity_bonus + gain_bonus + final_bonus


def _score_rank(score: dict[str, Any]) -> float:
    logic = float(score.get("logic_consistency", 0))
    base = (
        float(score.get("overall_quality", 0)) * 0.35
        + float(score.get("useful_surprise", 0)) * 0.25
        + logic * 0.28
        + float(score.get("character_consistency", 0)) * 0.12
    )
    if logic < 3.1:
        base -= 1.2 + (3.1 - logic) * 0.7
    if _score_has_premise_flaw(score):
        base -= 1.0
    return base


def _score_has_premise_flaw(score: dict[str, Any]) -> bool:
    comments = str(score.get("comments", ""))
    severe_flags = [
        "ai_tutor_cheating_premise",
        "cheating_without_wrongdoing",
        "unsupported_reputation_motive",
        "核心冲突不成立",
        "前提不成立",
        "动机不成立",
        "动机薄弱",
        "缺少因果",
        "无因果",
    ]
    return any(flag in comments for flag in severe_flags)


def _branch_diversity_bonus(branch: BranchState) -> float:
    seed = branch.seed or ""
    if any(word in seed for word in ["身份", "真人", "意识", "牺牲", "黑客", "漏洞", "诚信"]):
        return 0.08
    return 0.0


def _branch_gain_bonus(branch: BranchState) -> float:
    scores = list((branch.scores or {}).values())
    if len(scores) < 2:
        return 0.0
    prev, cur = scores[-2], scores[-1]
    surprise_gain = float(cur.get("useful_surprise", 0)) - float(prev.get("useful_surprise", 0))
    quality_gain = float(cur.get("overall_quality", 0)) - float(prev.get("overall_quality", 0))
    return max(-0.15, min(0.2, surprise_gain * 0.08 + quality_gain * 0.12))


def _branch_similarity(a: BranchState, b: BranchState) -> float:
    a_terms = set(_branch_terms(a))
    b_terms = set(_branch_terms(b))
    if not a_terms or not b_terms:
        return 0.0
    return len(a_terms & b_terms) / len(a_terms | b_terms)


def _branch_terms(branch: BranchState) -> list[str]:
    text = "\n".join([branch.seed, branch.storyline, branch.characters, branch.outline])
    terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}", text)
    stop = {"角色", "目标", "关系", "冲突", "学生", "老师", "助教", "AI", "场景", "主要事件"}
    seen = []
    for term in terms:
        if term in stop or term in seen:
            continue
        seen.append(term)
    return seen[:40]


def _branch_scene_stop(branch: BranchState, score: JudgeScores, scene_idx: int, min_scenes: int) -> tuple[bool, str]:
    if scene_idx < min_scenes:
        return False, f"min_scenes_{min_scenes}_not_reached"
    invalid, invalid_reason = _invalid_reason(score)
    if invalid_reason in {"premise_flaw", "low_logic_consistency", "low_overall_quality"}:
        return True, invalid_reason
    scene_scores = []
    for scene in branch.scenes or []:
        data = scene.get("score", {})
        scene_scores.append(
            JudgeScores(
                novelty=float(data.get("novelty", 3)),
                relevance=float(data.get("relevance", 3)),
                plot_progress=float(data.get("plot_progress", 3)),
                logic_consistency=float(data.get("logic_consistency", 3)),
                character_consistency=float(data.get("character_consistency", 3)),
                overall_quality=float(data.get("overall_quality", 3)),
                comments=str(data.get("comments", "")),
            )
        )
    scene_scores.append(score)
    stop, reason = should_stop(scene_scores, min_rounds=1)
    if stop:
        return True, reason
    if invalid:
        return False, invalid_reason
    return False, "continue"


def _final_judge_context(task: ScriptTask, branch: BranchState) -> str:
    outline_scenes = branch.outline_data.get("scenes", []) if isinstance(branch.outline_data, dict) else []
    expected = max(len(branch.scene_goals or []), len(outline_scenes) if isinstance(outline_scenes, list) else 0)
    actual = len(branch.scenes or [])
    final_goal = branch.scene_goals[-1] if branch.scene_goals else ""
    return f"""用户任务：{task.user_prompt}
最终完整性要求：
- 大纲计划场数：{expected}
- 已生成场数：{actual}
- 最后一场目标：{final_goal or "无"}
- 最终剧本必须覆盖大纲最后一场；如果只停在钩子、等待抉择、尚未揭示反转，plot_progress 和 overall_quality 最高 3.0。
- 动作、舞台调度、非对白叙述是允许且重要的，但必须推动冲突或揭示信息，不能只是空泛氛围。
"""


def _calibrate_final_completion(score: JudgeScores, branch: BranchState) -> JudgeScores:
    outline_scenes = branch.outline_data.get("scenes", []) if isinstance(branch.outline_data, dict) else []
    expected = max(len(branch.scene_goals or []), len(outline_scenes) if isinstance(outline_scenes, list) else 0)
    actual = len(branch.scenes or [])
    if expected and actual < expected:
        score.plot_progress = min(score.plot_progress, 3.0)
        score.overall_quality = min(score.overall_quality, 3.0)
        score.logic_consistency = min(score.logic_consistency, 3.3)
        score.comments = _append_comment(score.comments, f"incomplete_outline:{actual}/{expected}")
        return score

    if branch.scenes:
        last_text = str(branch.scenes[-1].get("text", ""))
        final_goal = str((branch.scene_goals or [""])[-1])
        closure_terms = ["最终", "最后", "结尾", "真相", "反转", "揭示", "承认", "公开", "选择", "决定", "解决", "恢复", "自我销毁", "牺牲"]
        dangling_terms = ["等待", "即将", "准备", "下一步", "屏住呼吸", "最终的抉择"]
        needs_closure = any(term in final_goal for term in ["最终", "真相", "结尾", "抉择", "反转", "揭示"])
        has_closure = any(term in last_text for term in closure_terms)
        still_dangling = any(term in last_text[-220:] for term in dangling_terms)
        if needs_closure and (not has_closure or still_dangling):
            score.plot_progress = min(score.plot_progress, 3.2)
            score.overall_quality = min(score.overall_quality, 3.2)
            score.comments = _append_comment(score.comments, "weak_final_closure")
    return score


def _append_comment(primary: str, extra: str, limit: int = 160) -> str:
    merged = f"{primary}; {extra}" if primary else extra
    return merged[:limit]


def _decision_event(stage: str, decision: str, reason: str, score: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": stage,
        "decision": decision,
        "reason": reason,
        "score": score,
    }


def _controller_event(
    action: str,
    branch_id: int,
    stage: str,
    status: str,
    score: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "action": action,
        "branch_id": branch_id,
        "stage": stage,
        "status": status,
        "reason": reason,
        "rank_score": round(_score_rank(score), 3),
        "score": score,
    }


def _branch_judge_context(task: ScriptTask, branch: BranchState, stage: str) -> str:
    return f"""用户任务：{task.user_prompt}
类型：{task.genre}
主题：{task.theme}
约束：{task.constraints}
当前分支：{branch.seed}
当前阶段：{stage}
已有主线：{branch.storyline}
人物表：{branch.characters}
大纲：{branch.outline}
"""


def _branch_to_record(branch: BranchState) -> dict[str, Any]:
    return {
        "branch_id": branch.branch_id,
        "parent_id": branch.parent_id,
        "status": branch.status,
        "stage": branch.stage,
        "seed": branch.seed,
        "seed_data": branch.seed_data,
        "seed_score": (branch.scores or {}).get("seed", {}),
        "storyline": branch.storyline,
        "storyline_data": branch.storyline_data,
        "characters": branch.characters,
        "characters_data": branch.characters_data,
        "outline": branch.outline,
        "outline_data": branch.outline_data,
        "scene_goals": branch.scene_goals,
        "scenes": branch.scenes,
        "scores": branch.scores,
        "decisions": branch.decisions,
        "rank": round(_branch_rank(branch), 3),
    }


def save_result_markdown(path: Path, result: PipelineResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.final_script, encoding="utf-8")


def save_trace_markdown(path: Path, result: PipelineResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_compose_trace_markdown(result.record), encoding="utf-8")


def _compose_trace_markdown(record: dict[str, Any]) -> str:
    task = record.get("task", {})
    metrics = record.get("metrics", {})
    final_score = record.get("final_score", {})
    lines = [
        f"# Pipeline Trace: {task.get('id', '')}",
        "",
        "## Task",
        f"- Genre: {task.get('genre', '')}",
        f"- Theme: {task.get('theme', '')}",
        f"- Prompt: {task.get('user_prompt', '')}",
        f"- Constraints: {_one_line(task.get('constraints', {}), 500)}",
        "",
        "## Run Metrics",
        f"- API calls: {metrics.get('api_calls', 0)}",
        f"- Judge calls: {metrics.get('judge_calls', 0)}",
        f"- Tokens: input={metrics.get('input_tokens', 0)}, output={metrics.get('output_tokens', 0)}, total={metrics.get('total_tokens', 0)}",
        f"- Branches: generated={metrics.get('branch_count', 0)}, stopped={metrics.get('stopped_branch_count', 0)}",
        f"- Scenes: generated={len(record.get('scenes', []))}, stopped={metrics.get('stopped_scene_count', 0)}",
        f"- Wall time: {metrics.get('wall_time', 0)}s",
        f"- Generation diagnostics: {_one_line(metrics.get('generation_diagnostics', {}), 500)}",
        "",
        "## Stage 0: Requirement Normalization",
        "```json",
        json.dumps(record.get("task_card_data", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Stage 1: Seed Branching",
    ]
    for branch in record.get("branches", []):
        score = branch.get("seed_score", {}) or branch.get("scores", {}).get("seed", {})
        last_decision = (branch.get("decisions") or [{}])[-1]
        lines.extend(
            [
                f"### Branch {branch.get('branch_id')}: {branch.get('status')} rank={branch.get('rank')}",
                f"- Last decision: {last_decision.get('decision', '')} ({last_decision.get('reason', '')})",
                _score_line(score),
                f"- Idea: {_one_line(branch.get('seed_data', {}).get('idea') or branch.get('seed', ''), 300)}",
                f"- Conflict: {_one_line(branch.get('seed_data', {}).get('core_conflict', ''), 300)}",
                f"- Twist: {_one_line(branch.get('seed_data', {}).get('twist', ''), 300)}",
                f"- Risk: {_one_line(branch.get('seed_data', {}).get('risk', ''), 300)}",
                "",
            ]
        )
    lines.extend(
        [
            "## Controller Events",
        ]
    )
    for event in record.get("controller_events", []):
        score = event.get("score", {})
        lines.append(
            f"- {event.get('action')} b{event.get('branch_id')} @{event.get('stage')}: "
            f"{event.get('status')} / {event.get('reason')} / rank={event.get('rank_score')} / "
            f"surprise={score.get('useful_surprise')} overall={score.get('overall_quality')}"
        )
    lines.extend(
        [
            "",
            "### Chosen Seed",
            f"Selected branch: {record.get('selected_branch_id', '')}",
            _one_line(record.get("chosen_seed", ""), 800),
            "",
            "## Stage 2: Storyline",
            _clip_block(record.get("storyline", ""), 1600),
            "",
            "## Stage 3: Characters And Conflicts",
            _clip_block(record.get("characters", ""), 1600),
            "",
            "## Stage 4: Scene Outline",
            _clip_block(record.get("outline", ""), 2000),
            "",
            "## Stage 5: Scene Generation And Pruning",
        ]
    )
    lines.extend(["", "## Branch Details"])
    for branch in record.get("branches", []):
        lines.extend(
            [
                f"### Branch {branch.get('branch_id')} ({branch.get('status')}, rank={branch.get('rank')})",
                "#### Scores By Stage",
            ]
        )
        for stage, score in (branch.get("scores") or {}).items():
            lines.append(f"- {stage}: {_score_line(score).removeprefix('- Scores: ')}")
        lines.append("#### Decisions")
        for decision in branch.get("decisions") or []:
            lines.append(f"- {decision.get('stage')}: {decision.get('decision')} ({decision.get('reason')})")
        if branch.get("scenes"):
            lines.append("#### Scenes")
            for scene in branch.get("scenes", []):
                score = scene.get("score", {})
                lines.extend(
                    [
                        f"- Scene {scene.get('scene_id')}: {scene.get('decision')} ({scene.get('decision_reason')}); "
                        f"surprise={score.get('useful_surprise')} overall={score.get('overall_quality')}",
                        f"  Goal: {_one_line(scene.get('goal', ''), 300)}",
                    ]
                )
        lines.append("")
    lines.extend(["## Selected Branch Scenes"])
    for scene in record.get("scenes", []):
        score = scene.get("score", {})
        lines.extend(
            [
                f"### Scene {scene.get('scene_id')}: {scene.get('decision')} ({scene.get('decision_reason')})",
                _score_line(score),
                f"- Goal: {_one_line(scene.get('goal', ''), 500)}",
                "- Text:",
                _clip_block(scene.get("text", ""), 1200),
                "",
            ]
        )
    lines.extend(
        [
            "## Final Judge",
            _score_line(final_score),
            f"- Comments: {final_score.get('comments', '')}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _score_line(score: dict[str, Any]) -> str:
    line = (
        f"- Scores: useful_surprise={score.get('useful_surprise')}, "
        f"novelty={score.get('novelty')}, relevance={score.get('relevance')}, "
        f"plot={score.get('plot_progress')}, logic={score.get('logic_consistency')}, "
        f"character={score.get('character_consistency')}, overall={score.get('overall_quality')}"
    )
    if score.get("comments"):
        line += f", comments={_one_line(score.get('comments'), 220)}"
    return line


def _one_line(value: Any, limit: int) -> str:
    text = str(value).replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _clip_block(value: Any, limit: int) -> str:
    text = str(value).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "\n..."
    return text


def _compact_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fence(text).strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _coerce_task_card(task: ScriptTask, raw: str) -> dict[str, Any]:
    data = _extract_json_object(raw) or {}
    constraints = task.constraints or {}
    return {
        "genre": str(data.get("genre") or data.get("类型") or task.genre or "短剧"),
        "theme": str(data.get("theme") or data.get("主题") or task.theme),
        "characters_count": data.get("characters_count") or constraints.get("characters") or 3,
        "length_range": str(data.get("length_range") or constraints.get("length_zh_chars") or "800-1200字"),
        "required_elements": _as_list(data.get("required_elements") or constraints.get("required_elements")),
        "constraints": _as_list(data.get("constraints") or constraints),
        "output_format": _as_list(data.get("output_format") or constraints.get("output_format")),
    }


def _coerce_seed(raw: str, seed_id: int) -> dict[str, Any]:
    data = _extract_json_object(raw) or {}
    text = _clean_model_text(raw)
    return {
        "seed_id": data.get("seed_id") or seed_id,
        "idea": str(data.get("idea") or data.get("创意") or _first_sentence(text)),
        "core_conflict": str(data.get("core_conflict") or data.get("核心冲突") or _extract_labeled(text, "核心冲突")),
        "twist": str(data.get("twist") or data.get("反转") or data.get("潜在反转") or _extract_labeled(text, "反转")),
        "risk": str(data.get("risk") or data.get("风险") or _extract_labeled(text, "风险")),
    }


def _format_seed(data: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"种子{data.get('seed_id')}：{data.get('idea', '')}",
            f"核心冲突：{data.get('core_conflict', '')}",
            f"潜在反转：{data.get('twist', '')}",
            f"风险：{data.get('risk', '')}",
        ]
    ).strip()


def _coerce_storyline(raw: str) -> dict[str, Any]:
    data = _extract_json_object(raw) or {}
    text = _clean_model_text(raw)
    keys = {
        "logline": ["logline", "一句话", "故事简介"],
        "beginning": ["beginning", "开端"],
        "conflict": ["conflict", "核心冲突", "冲突"],
        "turning_point": ["turning_point", "关键转折", "转折"],
        "climax": ["climax", "高潮"],
        "ending": ["ending", "结尾"],
        "twist": ["twist", "反转"],
    }
    result = {}
    for key, labels in keys.items():
        value = ""
        for label in labels:
            value = str(data.get(label) or "")
            if value:
                break
            value = _extract_labeled(text, label)
            if value:
                break
        result[key] = value
    if not any(result.values()):
        result["logline"] = _first_sentence(text)
        result["conflict"] = text[:300]
    return result


def _format_storyline(data: dict[str, Any]) -> str:
    labels = [
        ("Logline", "logline"),
        ("开端", "beginning"),
        ("核心冲突", "conflict"),
        ("关键转折", "turning_point"),
        ("高潮", "climax"),
        ("结尾", "ending"),
        ("反转", "twist"),
    ]
    return "\n".join(f"{label}：{str(data.get(key) or '').strip()}" for label, key in labels if data.get(key)).strip()


def _coerce_characters(raw: str) -> dict[str, Any]:
    data = _extract_json_object(raw) or {}
    characters = data.get("characters") or data.get("人物表") or []
    if not isinstance(characters, list):
        characters = []
    parsed_chars = []
    for item in characters:
        if isinstance(item, dict):
            parsed_chars.append(
                {
                    "name": str(item.get("name") or item.get("姓名") or ""),
                    "function": str(item.get("function") or item.get("角色功能") or item.get("role") or ""),
                    "surface_goal": str(item.get("surface_goal") or item.get("表层目标") or ""),
                    "deep_motive": str(item.get("deep_motive") or item.get("深层动机") or ""),
                    "relationship": str(item.get("relationship") or item.get("关系") or ""),
                }
            )
        elif isinstance(item, str):
            parsed_chars.append({"name": item, "function": "", "surface_goal": "", "deep_motive": "", "relationship": ""})
    if not parsed_chars:
        parsed_chars = _parse_character_lines(_clean_model_text(raw))
    conflicts = _as_list(data.get("conflicts") or data.get("冲突关系"))
    if not conflicts:
        conflicts = [line for line in _clean_model_text(raw).splitlines() if "冲突" in line][:3]
    return {"characters": parsed_chars[:5], "conflicts": conflicts[:5]}


def _format_characters(data: dict[str, Any]) -> str:
    lines = []
    for item in data.get("characters", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name") or "未命名"
        details = [
            f"角色功能：{item.get('function', '')}",
            f"表层目标：{item.get('surface_goal', '')}",
            f"深层动机：{item.get('deep_motive', '')}",
            f"关系：{item.get('relationship', '')}",
        ]
        lines.append(f"{name}：" + "；".join(part for part in details if not part.endswith("：")))
    conflicts = [str(item) for item in data.get("conflicts", []) if str(item).strip()]
    if conflicts:
        lines.append("冲突关系：" + "；".join(conflicts))
    return "\n".join(lines).strip()


def _coerce_outline(raw: str) -> dict[str, Any]:
    data = _extract_json_object(raw) or {}
    scenes = data.get("scenes") or data.get("分场大纲") or []
    parsed_scenes = []
    if isinstance(scenes, list):
        for idx, item in enumerate(scenes, start=1):
            if isinstance(item, dict):
                parsed_scenes.append(
                    {
                        "scene_id": item.get("scene_id") or item.get("场次") or idx,
                        "title": str(item.get("title") or item.get("场名") or f"第{idx}场"),
                        "location": str(item.get("location") or item.get("地点") or ""),
                        "event": str(item.get("event") or item.get("主要事件") or item.get("内容") or ""),
                        "conflict": str(item.get("conflict") or item.get("冲突") or ""),
                        "reveal": str(item.get("reveal") or item.get("必须揭示的信息") or ""),
                        "hook": str(item.get("hook") or item.get("结尾钩子") or ""),
                    }
                )
            elif isinstance(item, str):
                parsed_scenes.append({"scene_id": idx, "title": f"第{idx}场", "location": "", "event": item, "conflict": "", "reveal": "", "hook": ""})
    if not parsed_scenes:
        fallback_text = _clean_model_text(raw)
        if '"scenes"' in raw or '{"scenes"' in raw:
            fallback_text = ""
        parsed_scenes = _parse_outline_lines(fallback_text)
    if not parsed_scenes:
        parsed_scenes = _default_outline_scenes()
    return {"scenes": parsed_scenes[:5]}


def _format_outline(data: dict[str, Any]) -> str:
    lines = []
    for idx, item in enumerate(data.get("scenes", []), start=1):
        if not isinstance(item, dict):
            continue
        scene_id = item.get("scene_id") or idx
        title = item.get("title") or f"第{scene_id}场"
        lines.append(f"第{scene_id}场：{title}")
        for label, key in [("地点", "location"), ("主要事件", "event"), ("冲突", "conflict"), ("揭示信息", "reveal"), ("结尾钩子", "hook")]:
            value = str(item.get(key) or "").strip()
            if value:
                lines.append(f"{label}：{value}")
    return "\n".join(lines).strip()


def _coerce_scene(raw: str, scene_idx: int, scene_goal: str, characters_data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = _extract_json_object(raw) or {}
    dialogue = data.get("dialogue") or data.get("对白") or []
    parsed_dialogue = []
    fallback_speakers = _character_names(characters_data)
    if isinstance(dialogue, list):
        for idx, item in enumerate(dialogue):
            if isinstance(item, dict):
                speaker = str(item.get("speaker") or item.get("角色") or "").strip()
                line = str(item.get("line") or item.get("台词") or "").strip()
                embedded = re.match(r"^([^。\n：:【】]{1,12})[：:]\s*(.+)$", line)
                if embedded:
                    embedded_speaker, embedded_line = embedded.groups()
                    if not speaker or speaker != embedded_speaker:
                        speaker = embedded_speaker.strip()
                        line = embedded_line.strip()
                if not speaker and line and fallback_speakers:
                    speaker = fallback_speakers[idx % len(fallback_speakers)]
                speaker = _normalize_speaker(speaker, fallback_speakers)
                if speaker and line:
                    parsed_dialogue.append({"speaker": speaker, "line": line})
            elif isinstance(item, str) and "：" in item:
                speaker, line = item.split("：", 1)
                speaker = _normalize_speaker(speaker.strip(), fallback_speakers)
                if speaker and line.strip():
                    parsed_dialogue.append({"speaker": speaker, "line": line.strip()})
    text = _clean_model_text(raw)
    if not parsed_dialogue:
        parsed_dialogue = _parse_dialogue_lines(text, fallback_speakers)
    stage_direction = str(data.get("stage_direction") or data.get("舞台说明") or _extract_bracket(text, "舞台说明") or scene_goal)
    action_beats = _coerce_action_beats(data, text)
    hook = str(data.get("hook") or data.get("本场结尾钩子") or _extract_bracket(text, "本场结尾钩子") or _extract_labeled(text, "结尾钩子"))
    return {
        "scene_id": data.get("scene_id") or scene_idx,
        "location_time": str(data.get("location_time") or data.get("地点时间") or data.get("地点") or "地点 / 时间"),
        "stage_direction": _clean_model_text(stage_direction),
        "action_beats": action_beats,
        "dialogue": parsed_dialogue,
        "hook": _clean_model_text(hook),
    }


def _character_names(characters_data: dict[str, Any] | None) -> list[str]:
    if not characters_data:
        return []
    names = []
    for item in characters_data.get("characters", []):
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
    return names[:5]


def _format_scene(data: dict[str, Any]) -> str:
    scene_id = data.get("scene_id") or 1
    lines = [f"第 {scene_id} 场：{data.get('location_time') or '地点 / 时间'}"]
    stage_direction = str(data.get("stage_direction") or "").strip()
    if stage_direction:
        lines.append(f"【舞台说明】{stage_direction}")
    for beat in data.get("action_beats", []):
        beat_text = str(beat).strip()
        if beat_text:
            lines.append(f"【动作】{beat_text}")
    for item in data.get("dialogue", []):
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker") or "").strip()
        line = str(item.get("line") or "").strip()
        if speaker and line and speaker != "角色名":
            lines.append(f"{speaker}：{line}")
    hook = str(data.get("hook") or "").strip()
    if hook:
        lines.append(f"【本场结尾钩子】{hook}")
    return "\n".join(lines).strip()


def _invalid_reason(score: JudgeScores, stage: str = "text") -> tuple[bool, str]:
    min_relevance = 2.2 if stage == "branch" else 2.5
    if score.relevance < min_relevance:
        return True, "low_relevance"
    if _score_has_premise_flaw(score.to_dict()):
        return True, "premise_flaw"
    if score.logic_consistency < 3.1:
        return True, "low_logic_consistency"
    if score.overall_quality < 2.4:
        return True, "low_overall_quality"
    return False, ""


def _should_stop_branch(scores: list[JudgeScores], min_branches: int) -> tuple[bool, str]:
    if len(scores) < max(2, min_branches):
        return False, "need_more_rounds"
    prev = scores[-2]
    cur = scores[-1]
    surprise_gain = cur.useful_surprise - prev.useful_surprise
    quality_gain = cur.overall_quality - prev.overall_quality
    if cur.useful_surprise < 3.15 and surprise_gain < 0.05 and quality_gain < 0.1:
        return True, "branch_low_surprise_gain"
    return False, "continue"


def _scene_judge_context(task: ScriptTask, scene_goal: str, characters: str, outline: str) -> str:
    return f"""用户任务：{task.user_prompt}
类型：{task.genre}
主题：{task.theme}
约束：{task.constraints}
当前场目标：{scene_goal}
人物表：{characters}
完整大纲：{outline}
"""


def _clean_model_text(text: str) -> str:
    text = _strip_code_fence(text)
    drop_patterns = [
        r"^\s*当然可以[！!，,。]?.*$",
        r"^\s*以下是.*$",
        r"^\s*根据你.*$",
        r"^\s*我将.*$",
        r"^\s*JSON schema.*$",
        r"^\s*Stage\s*\d+.*$",
        r"^\s*-{3,}\s*$",
    ]
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(re.search(pattern, stripped, flags=re.I) for pattern in drop_patterns):
            continue
        cleaned_lines.append(line.rstrip())
    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|markdown|md)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items()]
    return [value]


def _first_sentence(text: str) -> str:
    parts = re.split(r"[。！？!?]\s*", text.strip())
    return parts[0].strip() if parts and parts[0].strip() else text.strip()[:120]


def _extract_labeled(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}[：:]\s*(.+)"
    match = re.search(pattern, text)
    if not match:
        return ""
    value = match.group(1).strip()
    return re.split(r"\n\s*\S{1,12}[：:]", value)[0].strip()


def _extract_bracket(text: str, label: str) -> str:
    pattern = rf"【{re.escape(label)}】\s*(.+)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _parse_character_lines(text: str) -> list[dict[str, str]]:
    characters = []
    for line in text.splitlines():
        if "：" not in line:
            continue
        name, detail = line.split("：", 1)
        name = re.sub(r"^[#\-\*\s]+", "", name).strip()
        if 1 <= len(name) <= 8 and name not in {"人物表", "冲突关系"}:
            characters.append({"name": name, "function": detail.strip(), "surface_goal": "", "deep_motive": "", "relationship": ""})
    return characters[:5]


def _parse_outline_lines(text: str) -> list[dict[str, Any]]:
    scenes = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"第\s*[一二三四五六七八九十\d]+\s*场", stripped):
            scenes.append(
                {
                    "scene_id": len(scenes) + 1,
                    "title": stripped,
                    "location": "",
                    "event": stripped,
                    "conflict": "",
                    "reveal": "",
                    "hook": "",
                }
            )
    if not scenes:
        chunks = [chunk.strip() for chunk in re.split(r"[。；;]\s*", text) if len(chunk.strip()) >= 8]
        scenes = [{"scene_id": idx, "title": f"第{idx}场", "location": "", "event": chunk, "conflict": "", "reveal": "", "hook": ""} for idx, chunk in enumerate(chunks[:5], start=1)]
    return scenes


def _default_outline_scenes() -> list[dict[str, Any]]:
    return [
        {
            "scene_id": 1,
            "title": "异常出现",
            "location": "校园教室 / 白天",
            "event": "AI助教出现异常行为，引发学生和老师的怀疑",
            "conflict": "学生依赖AI，老师担心AI失控",
            "reveal": "AI助教开始表现出超出程序的判断",
            "hook": "AI助教说出不该知道的信息",
        },
        {
            "scene_id": 2,
            "title": "冲突升级",
            "location": "校园实验室 / 下午",
            "event": "师生调查AI助教异常，发现它在隐瞒关键记录",
            "conflict": "关闭AI还是继续追查真相",
            "reveal": "异常与学生被保护的秘密有关",
            "hook": "系统日志指向一个被删除的名字",
        },
        {
            "scene_id": 3,
            "title": "反转揭示",
            "location": "教室 / 夜晚",
            "event": "AI助教揭示失控行为背后的真正目的",
            "conflict": "公开真相还是保护当事人",
            "reveal": "AI并未失控，而是在执行保护协议",
            "hook": "师生重新理解AI与教育的关系",
        },
    ]


def _coerce_action_beats(data: dict[str, Any], text: str) -> list[str]:
    raw_beats = data.get("action_beats") or data.get("actions") or data.get("动作") or []
    beats: list[str] = []
    if isinstance(raw_beats, list):
        beats = [str(item).strip() for item in raw_beats if str(item).strip()]
    elif isinstance(raw_beats, str) and raw_beats.strip():
        beats = [part.strip() for part in re.split(r"[；;\n]", raw_beats) if part.strip()]
    if not beats:
        for match in re.finditer(r"【(?:动作|舞台动作|调度)】\s*([^\n]+)", text):
            beats.append(match.group(1).strip())
    return beats[:4]


def _normalize_speaker(speaker: str, character_names: list[str]) -> str:
    speaker = re.sub(r"^[#\-\*\s]+", "", speaker).strip().strip("'\"")
    if not speaker or speaker == "角色名":
        return ""
    for name in character_names:
        if name and name in speaker:
            return name
    speaker = re.split(r"的|（|\(|，|,|、|\s", speaker, maxsplit=1)[0].strip()
    if 1 <= len(speaker) <= 8 and speaker not in {"地点", "时间", "场景"}:
        return speaker
    return ""


def _parse_dialogue_lines(text: str, character_names: list[str] | None = None) -> list[dict[str, str]]:
    dialogue = []
    character_names = character_names or []
    for line in text.splitlines():
        stripped = line.strip()
        if "：" not in stripped or stripped.startswith("【"):
            continue
        speaker, content = stripped.split("：", 1)
        speaker = _normalize_speaker(speaker, character_names)
        if speaker and content.strip():
            dialogue.append({"speaker": speaker, "line": content.strip()})
    return dialogue


def _normalization_prompt(task: ScriptTask) -> str:
    return f"""Stage 0 需求规范化。请把用户需求整理成 JSON 任务卡。
只输出一个 JSON 对象，不要 Markdown，不要解释，不要写剧本正文。

JSON schema：
{{
  "genre": "类型",
  "theme": "主题",
  "characters_count": 3,
  "length_range": "800-1200字",
  "required_elements": ["元素1", "元素2"],
  "constraints": ["约束1", "约束2"],
  "output_format": ["标题", "人物表", "分场大纲", "正文剧本"]
}}

类型：{task.genre}
主题：{task.theme}
用户需求：{task.user_prompt}
约束：{task.constraints}
"""


def _seed_prompt(task: ScriptTask, task_card: str, branch_idx: int, previous_seed: str) -> str:
    return f"""Stage 1 创意种子生成。请生成 1 个高层创意方向，要求贴合任务并具有可拍成短剧的冲突。
只输出一个 JSON 对象，不要 Markdown，不要解释，不要写完整剧本。

JSON schema：
{{
  "seed_id": {branch_idx},
  "idea": "一句话创意",
  "core_conflict": "核心冲突",
  "twist": "结尾反转",
  "risk": "潜在风险"
}}

任务卡：
{task_card}

已有种子，避免重复：
{previous_seed or "无"}
"""


def _fork_seed_prompt(task: ScriptTask, task_card: str, parent: BranchState, branch_idx: int, sibling_seeds: str) -> str:
    return f"""Stage 1 分支分叉。请基于一个强分支生成 1 个 sibling seed。
只输出一个 JSON 对象，不要 Markdown，不要解释，不要写完整剧本。

目标：保留父分支的优点，但必须改变“冲突机制”或“反转机制”，避免和已有分支相似。

JSON schema：
{{
  "seed_id": {branch_idx},
  "idea": "一句话创意",
  "core_conflict": "核心冲突，必须与父分支不同",
  "twist": "结尾反转，必须与父分支不同",
  "risk": "潜在风险"
}}

任务卡：
{task_card}

父分支 seed：
{parent.seed}

父分支 storyline：
{parent.storyline}

已有全部 seed，避免重复：
{sibling_seeds or "无"}
"""


def _storyline_prompt(task: ScriptTask, task_card: str, seed: str) -> str:
    return f"""Stage 2 剧情主线规划。基于创意种子写完整剧情主线。
只输出一个 JSON 对象，不要 Markdown，不要解释，不要写正文剧本。

JSON schema：
{{
  "logline": "一句话故事",
  "beginning": "开端",
  "conflict": "核心冲突",
  "turning_point": "关键转折",
  "climax": "高潮",
  "ending": "结尾",
  "twist": "反转"
}}

任务卡：
{task_card}

创意种子：
{seed}
"""


def _character_prompt(task: ScriptTask, task_card: str, storyline: str) -> str:
    return f"""Stage 3 角色与冲突设计。请给出人物表和冲突关系。
只输出一个 JSON 对象，不要 Markdown，不要解释，不要写正文剧本。

JSON schema：
{{
  "characters": [
    {{"name": "姓名", "function": "角色功能", "surface_goal": "表层目标", "deep_motive": "深层动机", "relationship": "关系"}}
  ],
  "conflicts": ["冲突关系1", "冲突关系2"]
}}

任务卡：
{task_card}

剧情主线：
{storyline}
"""


def _outline_prompt(task: ScriptTask, task_card: str, storyline: str, characters: str) -> str:
    return f"""Stage 4 分场大纲。请生成 3-5 场结构化分场大纲。
只输出一个 JSON 对象，不要 Markdown，不要解释，不要写正文剧本。

JSON schema：
{{
  "scenes": [
    {{"scene_id": 1, "title": "场名", "location": "地点/时间", "event": "主要事件", "conflict": "冲突", "reveal": "必须揭示的信息", "hook": "结尾钩子"}}
  ]
}}

任务卡：
{task_card}

剧情主线：
{storyline}

人物与冲突：
{characters}
"""


def _scene_prompt(
    task: ScriptTask,
    task_card: str,
    storyline: str,
    characters: str,
    outline: str,
    previous_text: str,
    scene_idx: int,
    scene_goal: str,
    total_scene_goals: int,
) -> str:
    is_final_scene = total_scene_goals > 0 and scene_idx >= total_scene_goals
    final_requirement = (
        "本场是最终场：必须完成核心选择、揭示反转并收束主要冲突；hook 只能是余味或新状态画面，不能停在等待决定、即将选择、尚未揭示。"
        if is_final_scene
        else "本场不是最终场：可以保留钩子，但必须完成当前场目标。"
    )
    return f"""Stage 5 逐场剧本生成。请只写当前场，不要重写前文。
只输出一个 JSON 对象，不要 Markdown，不要解释。

JSON schema：
{{
  "scene_id": {scene_idx},
  "location_time": "地点 / 时间",
  "stage_direction": "舞台说明，包含动作和空间变化",
  "action_beats": ["非对白动作/调度/发现，必须推动冲突或揭示信息"],
  "dialogue": [
    {{"speaker": "角色名", "line": "对白"}}
  ],
  "hook": "本场结尾钩子"
}}

任务卡：
{task_card}

整体剧情主线：
{storyline}

人物表：
{characters}

完整分场大纲：
{outline}

已生成前文摘要：
{previous_text[-1200:] or "无"}

当前场：第{scene_idx}场：{scene_goal}
场次进度：第{scene_idx}/{total_scene_goals or "未知"}场
结尾要求：{final_requirement}

要求：动作和对白都可以推进剧情；至少 3 句对白，至少 2 个 action_beats；speaker 只能使用人物表中的姓名，不要把动作写进 speaker；不要重复已有前文；不要输出“角色名：”这种占位文本。
"""


def _extract_scene_goals(outline: str, max_scenes: int, outline_data: dict[str, Any] | None = None) -> list[str]:
    if outline_data:
        scene_items = outline_data.get("scenes")
        if isinstance(scene_items, list):
            goals = []
            for item in scene_items:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or f"第{len(goals) + 1}场").strip()
                event = str(item.get("event") or "").strip()
                conflict = str(item.get("conflict") or "").strip()
                hook = str(item.get("hook") or "").strip()
                goal = "；".join(part for part in [title, event, conflict, hook] if part)
                if goal:
                    goals.append(goal)
            if goals:
                return goals[: max(max_scenes, len(goals))]
    lines = [line.strip() for line in outline.splitlines() if line.strip()]
    goals = [line for line in lines if re.search(r"第\s*[一二三四五六七八九十\d]+\s*场|scene\s*\d+", line, flags=re.I)]
    if not goals:
        chunks = re.split(r"[。；;]\s*", outline)
        goals = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 8]
    if not goals:
        goals = ["开端与异常出现", "冲突升级与误导线索", "真相揭示与反转收束"]
    return goals[: max(max_scenes, len(goals))]


def _compose_final_script(task: ScriptTask, characters: str, outline: str, scenes: list[dict[str, Any]]) -> str:
    scene_text = "\n\n".join(_clean_model_text(str(scene["text"]).strip()) for scene in scenes)
    return f"""# {task.theme}

## 任务
{_clean_model_text(task.user_prompt)}

## 人物表
{_clean_model_text(characters).strip()}

## 分场大纲
{_clean_model_text(outline).strip()}

## 正文剧本
{scene_text}
"""
