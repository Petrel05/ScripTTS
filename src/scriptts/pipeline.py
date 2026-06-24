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
    max_scenes: int = 4
    min_scenes: int = 3
    max_new_tokens: int = 768
    scene_max_new_tokens: int = 768
    temperature: float = 0.7
    judge_backend: str = "rule"


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
            "judge_calls": self.judge_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "branch_count": self.branch_count,
            "stopped_branch_count": self.stopped_branch_count,
            "stopped_scene_count": self.stopped_scene_count,
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

        raw_outputs: dict[str, Any] = {}

        task_card_raw = call(_normalization_prompt(task), temperature=0.2)
        raw_outputs["task_card"] = task_card_raw
        task_card_data = _coerce_task_card(task, task_card_raw)
        task_card = _compact_json(task_card_data)

        branch_records: list[dict[str, Any]] = []
        best_branch: dict[str, Any] | None = None
        previous_seed = ""
        recent_scores: list[JudgeScores] = []

        for idx in range(self.config.max_branches):
            seed_raw = call(_seed_prompt(task, task_card, idx + 1, previous_seed))
            seed_data = _coerce_seed(seed_raw, idx + 1)
            seed = _format_seed(seed_data)
            score = self._judge(seed, task.user_prompt, previous_seed, stats)
            recent_scores.append(score)
            invalid, invalid_reason = _invalid_reason(score, stage="branch")
            stop, reason = should_stop(recent_scores, min_rounds=1)
            branch_stop, branch_reason = _should_stop_branch(recent_scores)
            if branch_stop:
                stop = True
                reason = branch_reason
            if invalid:
                decision = "reject"
                reason = invalid_reason
            else:
                decision = "stop" if stop else "continue"
            stats.branch_count += 1
            branch = {
                "branch_id": idx + 1,
                "raw_seed": seed_raw,
                "seed_data": seed_data,
                "seed": seed,
                "seed_score": score.to_dict(),
                "decision": decision,
                "decision_reason": reason,
            }
            branch_records.append(branch)
            if not invalid and (best_branch is None or score.overall_quality + score.useful_surprise > (
                best_branch["score"].overall_quality + best_branch["score"].useful_surprise
            )):
                best_branch = {"seed": seed, "score": score, "data": seed_data}
            previous_seed += "\n" + seed
            if stop:
                stats.stopped_branch_count += 1
                break

        chosen_seed = str((best_branch or {"seed": branch_records[-1]["seed"]})["seed"])
        storyline_raw = call(_storyline_prompt(task, task_card, chosen_seed))
        raw_outputs["storyline"] = storyline_raw
        storyline_data = _coerce_storyline(storyline_raw)
        storyline = _format_storyline(storyline_data)

        characters_raw = call(_character_prompt(task, task_card, storyline))
        raw_outputs["characters"] = characters_raw
        characters_data = _coerce_characters(characters_raw)
        characters = _format_characters(characters_data)

        outline_raw = call(_outline_prompt(task, task_card, storyline, characters))
        raw_outputs["outline"] = outline_raw
        outline_data = _coerce_outline(outline_raw)
        outline = _format_outline(outline_data)
        scene_goals = _extract_scene_goals(outline, self.config.max_scenes, outline_data)

        scenes: list[dict[str, Any]] = []
        generated_so_far = ""
        scene_scores: list[JudgeScores] = []
        for scene_idx, scene_goal in enumerate(scene_goals, start=1):
            scene_raw = call(
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
            scene_data = _coerce_scene(scene_raw, scene_idx, scene_goal, characters_data)
            scene_text = _format_scene(scene_data)
            score = self._judge(scene_text, task.user_prompt, generated_so_far, stats)
            scene_scores.append(score)
            invalid, invalid_reason = _invalid_reason(score)
            stop, reason = should_stop(scene_scores, min_rounds=1)
            if scene_idx < self.config.min_scenes:
                stop = False
                if reason != "need_more_rounds":
                    reason = f"min_scenes_{self.config.min_scenes}_not_reached"
            decision = "accept"
            if invalid:
                decision = "accept_with_warning"
                reason = invalid_reason
                if scene_idx >= self.config.min_scenes:
                    decision = "accept_and_stop"
                    stop = True
            elif stop:
                decision = "accept_and_stop"
            scenes.append(
                {
                    "scene_id": scene_idx,
                    "goal": scene_goal,
                    "raw_text": scene_raw,
                    "scene_data": scene_data,
                    "text": scene_text,
                    "score": score.to_dict(),
                    "decision": decision,
                    "decision_reason": reason,
                }
            )
            generated_so_far = (generated_so_far + "\n\n" + scene_text).strip()
            if stop:
                stats.stopped_scene_count += 1
                break

        final_script = _compose_final_script(task, characters, outline, scenes)
        final_score = self._judge(final_script, task.user_prompt, "", stats)
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
            "branches": branch_records,
            "chosen_seed": chosen_seed,
            "storyline": storyline,
            "storyline_data": storyline_data,
            "characters": characters,
            "characters_data": characters_data,
            "outline": outline,
            "outline_data": outline_data,
            "scenes": scenes,
            "final_score": final_score.to_dict(),
            "metrics": stats.to_dict(),
            "raw_outputs": raw_outputs,
        }
        return PipelineResult(task_id=task.id, final_script=final_script, record=record)

    def _judge(self, candidate: str, task_prompt: str, previous_text: str, stats: PipelineStats) -> JudgeScores:
        if self.config.judge_backend == "llm":
            prompt = build_llm_judge_prompt(candidate, task_prompt, previous_text)
            generation = self.llm.generate(prompt, max_new_tokens=512, temperature=0.0)
            stats.add_generation(generation.input_tokens, generation.output_tokens)
            stats.judge_calls += 1
            return parse_llm_judge_response(generation.text, candidate, task_prompt, previous_text)
        return rule_judge(candidate, task_prompt, previous_text)


def save_result_markdown(path: Path, result: PipelineResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.final_script, encoding="utf-8")


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
        parsed_scenes = _parse_outline_lines(_clean_model_text(raw))
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
                if not speaker and line and fallback_speakers:
                    speaker = fallback_speakers[idx % len(fallback_speakers)]
                if speaker and line:
                    parsed_dialogue.append({"speaker": speaker, "line": line})
            elif isinstance(item, str) and "：" in item:
                speaker, line = item.split("：", 1)
                parsed_dialogue.append({"speaker": speaker.strip(), "line": line.strip()})
    text = _clean_model_text(raw)
    if not parsed_dialogue:
        parsed_dialogue = _parse_dialogue_lines(text)
    stage_direction = str(data.get("stage_direction") or data.get("舞台说明") or _extract_bracket(text, "舞台说明") or scene_goal)
    hook = str(data.get("hook") or data.get("本场结尾钩子") or _extract_bracket(text, "本场结尾钩子") or _extract_labeled(text, "结尾钩子"))
    return {
        "scene_id": data.get("scene_id") or scene_idx,
        "location_time": str(data.get("location_time") or data.get("地点时间") or data.get("地点") or "地点 / 时间"),
        "stage_direction": _clean_model_text(stage_direction),
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
    if score.logic_consistency < 2.8:
        return True, "low_logic_consistency"
    if score.overall_quality < 2.4:
        return True, "low_overall_quality"
    return False, ""


def _should_stop_branch(scores: list[JudgeScores]) -> tuple[bool, str]:
    if len(scores) < 2:
        return False, "need_more_rounds"
    prev = scores[-2]
    cur = scores[-1]
    surprise_gain = cur.useful_surprise - prev.useful_surprise
    quality_gain = cur.overall_quality - prev.overall_quality
    if cur.useful_surprise < 3.15 and surprise_gain < 0.05 and quality_gain < 0.1:
        return True, "branch_low_surprise_gain"
    return False, "continue"


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


def _parse_dialogue_lines(text: str) -> list[dict[str, str]]:
    dialogue = []
    for line in text.splitlines():
        stripped = line.strip()
        if "：" not in stripped or stripped.startswith("【"):
            continue
        speaker, content = stripped.split("：", 1)
        speaker = re.sub(r"^[#\-\*\s]+", "", speaker).strip()
        if 1 <= len(speaker) <= 12 and content.strip() and speaker not in {"地点", "时间", "场景", "角色名"}:
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
) -> str:
    return f"""Stage 5 逐场剧本生成。请只写当前场，不要重写前文。
只输出一个 JSON 对象，不要 Markdown，不要解释。

JSON schema：
{{
  "scene_id": {scene_idx},
  "location_time": "地点 / 时间",
  "stage_direction": "舞台说明，包含动作和空间变化",
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

要求：至少 6 句对白；必须有动作推进；不要重复已有前文；不要输出“角色名：”这种占位文本。
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
                return goals[:max_scenes]
    lines = [line.strip() for line in outline.splitlines() if line.strip()]
    goals = [line for line in lines if re.search(r"第\s*[一二三四五六七八九十\d]+\s*场|scene\s*\d+", line, flags=re.I)]
    if not goals:
        chunks = re.split(r"[。；;]\s*", outline)
        goals = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 8]
    if not goals:
        goals = ["开端与异常出现", "冲突升级与误导线索", "真相揭示与反转收束"]
    return goals[:max_scenes]


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
