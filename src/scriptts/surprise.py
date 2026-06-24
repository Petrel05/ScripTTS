from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .llm import LLM


@dataclass
class JudgeScores:
    novelty: float
    relevance: float
    plot_progress: float
    logic_consistency: float
    character_consistency: float
    overall_quality: float
    comments: str = ""

    @property
    def useful_surprise(self) -> float:
        return 0.35 * self.novelty + 0.35 * self.plot_progress + 0.30 * self.relevance

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["useful_surprise"] = round(self.useful_surprise, 3)
        return data


def rule_judge(candidate: str, task_prompt: str = "", previous_text: str = "") -> JudgeScores:
    text = candidate.strip()
    prompt_terms = _content_terms(task_prompt)
    relevance_hits = sum(1 for term in prompt_terms if term in text)
    relevance = _clamp(2.4 + min(relevance_hits, 6) * 0.35)

    surprise_markers = ["反转", "转折", "真相", "误会", "秘密", "异常", "发现", "揭示", "选择"]
    progress_markers = ["因此", "于是", "却", "但是", "最后", "必须", "冲突", "证据", "决定", "质问", "阻止"]
    repetition_penalty = _repetition_penalty(text)
    artifact_penalty = _artifact_penalty(text)
    similarity_penalty = _similarity_penalty(text, previous_text)
    novelty = _clamp(2.7 + _marker_score(text, surprise_markers) - similarity_penalty - repetition_penalty)
    plot_progress = _clamp(2.6 + _marker_score(text, progress_markers) - repetition_penalty)

    has_dialogue = bool(re.search(r"^[^。\n]{1,12}[：:]", text, flags=re.MULTILINE))
    has_scene = bool(re.search(r"第\s*[一二三四五六七八九十\d]+\s*场|【舞台说明】|人物表|分场大纲", text))
    has_hook = "钩子" in text or "反转" in text or "结尾" in text
    has_generic_role_label = "角色名：" in text or "speaker" in text.lower()
    structure_bonus = (0.45 if has_dialogue else -0.45) + (0.35 if has_scene else -0.25) + (0.2 if has_hook else 0)
    length_score = min(len(text) / 1400, 1.0) * 0.55
    overall = _clamp(2.7 + structure_bonus + length_score - artifact_penalty - repetition_penalty)

    contradiction_words = ["前后矛盾", "无法解释", "突然", "莫名其妙", "无缘无故", "没有原因"]
    logic = _clamp(4.1 - 0.35 * sum(1 for word in contradiction_words if word in text) - repetition_penalty * 0.4)
    speakers = _speakers(text)
    speaker_bonus = 0.25 if 2 <= len(speakers) <= 5 else -0.2
    char_consistency = _clamp(3.45 + (0.25 if has_dialogue else -0.25) + speaker_bonus - (0.35 if has_generic_role_label else 0))
    comments = (
        "rule_judge: readability-aware heuristic; "
        f"terms={relevance_hits}, repetition_penalty={repetition_penalty:.2f}, artifact_penalty={artifact_penalty:.2f}."
    )

    return JudgeScores(
        novelty=novelty,
        relevance=relevance,
        plot_progress=plot_progress,
        logic_consistency=logic,
        character_consistency=char_consistency,
        overall_quality=overall,
        comments=comments,
    )


def llm_judge(llm: LLM, candidate: str, task_prompt: str, previous_text: str = "", max_new_tokens: int = 512) -> JudgeScores:
    prompt = build_llm_judge_prompt(candidate, task_prompt, previous_text)
    raw = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0).text
    return parse_llm_judge_response(raw, candidate, task_prompt, previous_text)


def build_llm_judge_prompt(candidate: str, task_prompt: str, previous_text: str = "") -> str:
    return f"""你是剧本生成 pipeline 的严格评审器。只输出一行 JSON，不要 Markdown，不要解释。

评分必须是 1 到 5 的数字，可以有一位小数。不要因为出现关键词就给高分，要惩罚：
1. 助手口吻，如“当然可以”“以下是”“根据你的要求”。
2. 重复对白、重复场景、原地解释。
3. JSON/Markdown 说明文字混入最终剧本。
4. 缺少对白、缺少场景动作、人物目标不清。

JSON schema：
{{"novelty":3.0,"relevance":3.0,"plot_progress":3.0,"logic_consistency":3.0,"character_consistency":3.0,"overall_quality":3.0,"comments":"20字以内中文短评"}}

用户任务：
{task_prompt}

已有文本摘要：
{previous_text[-1200:] or "无"}

待评价候选：
{candidate[-3000:]}
"""


def parse_llm_judge_response(
    raw: str,
    candidate: str,
    task_prompt: str,
    previous_text: str = "",
) -> JudgeScores:
    data = _extract_json(raw)
    if not data:
        return rule_judge(candidate, task_prompt, previous_text)
    return JudgeScores(
        novelty=_clamp(float(data.get("novelty", 3))),
        relevance=_clamp(float(data.get("relevance", 3))),
        plot_progress=_clamp(float(data.get("plot_progress", data.get("plotProgress", 3)))),
        logic_consistency=_clamp(float(data.get("logic_consistency", data.get("logicConsistency", 3)))),
        character_consistency=_clamp(float(data.get("character_consistency", data.get("characterConsistency", 3)))),
        overall_quality=_clamp(float(data.get("overall_quality", data.get("overallQuality", 3)))),
        comments=str(data.get("comments", "llm_judge")),
    )


def should_stop(recent_scores: list[JudgeScores], min_rounds: int = 2) -> tuple[bool, str]:
    if len(recent_scores) < min_rounds + 1:
        return False, "need_more_rounds"
    tail = recent_scores[-min_rounds:]
    prev = recent_scores[-min_rounds - 1 : -1]
    low_surprise = all(score.useful_surprise < 3.2 for score in tail)
    low_gain = all((tail[i].overall_quality - prev[i].overall_quality) < 0.2 for i in range(min_rounds))
    valid = all(score.relevance >= 3 and score.logic_consistency >= 3 for score in tail)
    if low_surprise and low_gain and valid:
        return True, "low_surprise_and_low_quality_gain"
    return False, "continue"


def _extract_json(text: str) -> dict[str, Any] | None:
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _content_terms(text: str) -> list[str]:
    terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}", text)
    stop = {"要求", "主题", "一个", "人物", "结尾", "短剧", "剧本", "需要", "output", "format"}
    seen: list[str] = []
    for term in terms:
        if term in stop or term in seen:
            continue
        seen.append(term)
    return seen[:12]


def _marker_score(text: str, markers: list[str]) -> float:
    hits = sum(1 for marker in markers if marker in text)
    return min(hits, 5) * 0.28


def _similarity_penalty(text: str, previous_text: str) -> float:
    if not text or not previous_text:
        return 0.0
    a = set(_content_terms(text))
    b = set(_content_terms(previous_text))
    if not a or not b:
        return 0.0
    jaccard = len(a & b) / len(a | b)
    return min(0.9, jaccard * 1.2)


def _artifact_penalty(text: str) -> float:
    artifacts = [
        "当然可以",
        "以下是",
        "根据你的",
        "我将",
        "Stage",
        "JSON",
        "```",
        "schema",
        "任务卡",
    ]
    return min(1.2, sum(0.22 for artifact in artifacts if artifact in text))


def _repetition_penalty(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    duplicate_lines = len(lines) - len(set(lines))
    line_penalty = min(0.8, duplicate_lines / max(len(lines), 1) * 1.8)
    sentences = [part.strip() for part in re.split(r"[。！？!?]\s*", text) if len(part.strip()) >= 6]
    duplicate_sentences = len(sentences) - len(set(sentences))
    sentence_penalty = min(0.6, duplicate_sentences / max(len(sentences), 1) * 1.5)
    return round(min(1.1, line_penalty + sentence_penalty), 2)


def _speakers(text: str) -> set[str]:
    names = set()
    for match in re.finditer(r"^([^。\n：:【】]{1,12})[：:]", text, flags=re.MULTILINE):
        name = match.group(1).strip()
        if name and name not in {"地点", "时间", "场景", "角色名"}:
            names.add(name)
    return names


def _clamp(value: float) -> float:
    return round(max(1.0, min(5.0, value)), 2)
