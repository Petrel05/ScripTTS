from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .data import ScriptTask
from .llm import LLM
from .surprise import JudgeScores, llm_judge, rule_judge, should_stop


@dataclass
class PipelineConfig:
    max_branches: int = 2
    max_scenes: int = 4
    max_new_tokens: int = 768
    scene_max_new_tokens: int = 768
    temperature: float = 0.7
    judge_backend: str = "rule"


@dataclass
class PipelineStats:
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    branch_count: int = 0
    stopped_branch_count: int = 0
    wall_time: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add_generation(self, generation_input_tokens: int, generation_output_tokens: int) -> None:
        self.api_calls += 1
        self.input_tokens += generation_input_tokens
        self.output_tokens += generation_output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "branch_count": self.branch_count,
            "stopped_branch_count": self.stopped_branch_count,
            "wall_time": round(self.wall_time, 3),
        }


@dataclass
class PipelineResult:
    task_id: str
    final_script: str
    record: dict[str, Any]


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
            stats.add_generation(generation.input_tokens, generation.output_tokens)
            return generation.text.strip()

        task_card = call(_normalization_prompt(task), temperature=0.2)

        branch_records: list[dict[str, Any]] = []
        best_branch: dict[str, Any] | None = None
        previous_seed = ""
        recent_scores: list[JudgeScores] = []

        for idx in range(self.config.max_branches):
            seed = call(_seed_prompt(task, task_card, idx + 1, previous_seed))
            score = self._judge(seed, task.user_prompt, previous_seed)
            recent_scores.append(score)
            stop, reason = should_stop(recent_scores)
            stats.branch_count += 1
            branch = {
                "branch_id": idx + 1,
                "seed": seed,
                "seed_score": score.to_dict(),
                "decision": "stop" if stop else "continue",
                "decision_reason": reason,
            }
            branch_records.append(branch)
            if best_branch is None or score.overall_quality + score.useful_surprise > (
                best_branch["score"].overall_quality + best_branch["score"].useful_surprise
            ):
                best_branch = {"seed": seed, "score": score}
            previous_seed += "\n" + seed
            if stop:
                stats.stopped_branch_count += 1
                break

        chosen_seed = str((best_branch or {"seed": branch_records[-1]["seed"]})["seed"])
        storyline = call(_storyline_prompt(task, task_card, chosen_seed))
        characters = call(_character_prompt(task, task_card, storyline))
        outline = call(_outline_prompt(task, task_card, storyline, characters))
        scene_goals = _extract_scene_goals(outline, self.config.max_scenes)

        scenes: list[dict[str, Any]] = []
        generated_so_far = ""
        scene_scores: list[JudgeScores] = []
        for scene_idx, scene_goal in enumerate(scene_goals, start=1):
            scene_text = call(
                _scene_prompt(
                    task=task,
                    task_card=task_card,
                    storyline=storyline,
                    characters=characters,
                    outline=outline,
                    previous_text=generated_so_far,
                    scene_idx=scene_idx,
                    scene_goal=scene_goal,
                ),
                max_new_tokens=self.config.scene_max_new_tokens,
            )
            score = self._judge(scene_text, task.user_prompt, generated_so_far)
            scene_scores.append(score)
            stop, reason = should_stop(scene_scores)
            scenes.append(
                {
                    "scene_id": scene_idx,
                    "goal": scene_goal,
                    "text": scene_text,
                    "score": score.to_dict(),
                    "decision": "stop_current_scene" if stop else "accept",
                    "decision_reason": reason,
                }
            )
            generated_so_far = (generated_so_far + "\n\n" + scene_text).strip()

        final_script = _compose_final_script(task, characters, outline, scenes)
        final_score = self._judge(final_script, task.user_prompt, "")
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
            "branches": branch_records,
            "chosen_seed": chosen_seed,
            "storyline": storyline,
            "characters": characters,
            "outline": outline,
            "scenes": scenes,
            "final_score": final_score.to_dict(),
            "metrics": stats.to_dict(),
        }
        return PipelineResult(task_id=task.id, final_script=final_script, record=record)

    def _judge(self, candidate: str, task_prompt: str, previous_text: str) -> JudgeScores:
        if self.config.judge_backend == "llm":
            return llm_judge(self.llm, candidate, task_prompt, previous_text)
        return rule_judge(candidate, task_prompt, previous_text)


def save_result_markdown(path: Path, result: PipelineResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.final_script, encoding="utf-8")


def _normalization_prompt(task: ScriptTask) -> str:
    return f"""Stage 0 需求规范化。请把用户需求整理成 JSON 任务卡，只输出 JSON。

类型：{task.genre}
主题：{task.theme}
用户需求：{task.user_prompt}
约束：{task.constraints}
"""


def _seed_prompt(task: ScriptTask, task_card: str, branch_idx: int, previous_seed: str) -> str:
    return f"""Stage 1 创意种子生成。请生成 1 个高层创意方向，要求贴合任务并具有可拍成短剧的冲突。

任务卡：
{task_card}

已有种子，避免重复：
{previous_seed or "无"}

请输出：种子{branch_idx}、核心冲突、潜在反转、风险。
"""


def _storyline_prompt(task: ScriptTask, task_card: str, seed: str) -> str:
    return f"""Stage 2 剧情主线规划。基于创意种子写完整剧情主线。

任务卡：
{task_card}

创意种子：
{seed}

请包含 logline、开端、核心冲突、关键转折、高潮、结尾与反转。
"""


def _character_prompt(task: ScriptTask, task_card: str, storyline: str) -> str:
    return f"""Stage 3 角色与冲突设计。请给出人物表和冲突关系。

任务卡：
{task_card}

剧情主线：
{storyline}

每个主要角色包含：姓名、角色功能、表层目标、深层动机、关系。
"""


def _outline_prompt(task: ScriptTask, task_card: str, storyline: str, characters: str) -> str:
    return f"""Stage 4 分场大纲。请生成 3-5 场结构化分场大纲。

任务卡：
{task_card}

剧情主线：
{storyline}

人物与冲突：
{characters}

每场包含：地点、主要事件、冲突、必须揭示的信息、结尾钩子。
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
) -> str:
    return f"""Stage 5 逐场剧本生成。请只写当前场，不要重写前文。

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

格式要求：
第 {scene_idx} 场：地点 / 时间
【舞台说明】...
角色名：对白
【本场结尾钩子】...
"""


def _extract_scene_goals(outline: str, max_scenes: int) -> list[str]:
    lines = [line.strip() for line in outline.splitlines() if line.strip()]
    goals = [line for line in lines if re.search(r"第\s*[一二三四五六七八九十\d]+\s*场|scene\s*\d+", line, flags=re.I)]
    if not goals:
        chunks = re.split(r"[。；;]\s*", outline)
        goals = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 8]
    if not goals:
        goals = ["开端与异常出现", "冲突升级与误导线索", "真相揭示与反转收束"]
    return goals[:max_scenes]


def _compose_final_script(task: ScriptTask, characters: str, outline: str, scenes: list[dict[str, Any]]) -> str:
    scene_text = "\n\n".join(str(scene["text"]).strip() for scene in scenes)
    return f"""# {task.theme}

## 任务
{task.user_prompt}

## 人物表
{characters.strip()}

## 分场大纲
{outline.strip()}

## 正文剧本
{scene_text}
"""
