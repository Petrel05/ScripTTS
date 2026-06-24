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
    relevance = _clamp(3.0 + min(relevance_hits, 4) * 0.35)

    surprise_markers = ["反转", "转折", "真相", "误会", "秘密", "异常", "发现", "揭示", "选择"]
    progress_markers = ["因此", "于是", "却", "但是", "最后", "必须", "冲突", "证据", "决定"]
    novelty = _clamp(2.8 + _marker_score(text, surprise_markers) - _similarity_penalty(text, previous_text))
    plot_progress = _clamp(2.8 + _marker_score(text, progress_markers))

    has_dialogue = bool(re.search(r"^[^。\n]{1,12}[：:]", text, flags=re.MULTILINE))
    has_scene = "场" in text or "舞台说明" in text or "人物表" in text
    overall = _clamp(2.8 + (0.45 if has_dialogue else 0) + (0.35 if has_scene else 0) + min(len(text) / 1200, 1.0) * 0.6)

    contradiction_words = ["前后矛盾", "无法解释", "突然", "莫名其妙"]
    logic = _clamp(4.0 - 0.35 * sum(1 for word in contradiction_words if word in text))
    char_consistency = _clamp(3.7 + (0.3 if has_dialogue else 0))

    return JudgeScores(
        novelty=novelty,
        relevance=relevance,
        plot_progress=plot_progress,
        logic_consistency=logic,
        character_consistency=char_consistency,
        overall_quality=overall,
        comments="rule_judge: lightweight heuristic scores for pipeline smoke testing.",
    )


def llm_judge(llm: LLM, candidate: str, task_prompt: str, previous_text: str = "", max_new_tokens: int = 512) -> JudgeScores:
    prompt = f"""你是剧本生成实验的 judge。请只输出 JSON，不要解释。

评分范围 1-5，字段必须包含：
novelty, relevance, plot_progress, logic_consistency, character_consistency, overall_quality, comments。

用户任务：
{task_prompt}

已有文本摘要：
{previous_text[-1200:]}

待评价候选：
{candidate[-3000:]}
"""
    raw = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0).text
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


def _clamp(value: float) -> float:
    return round(max(1.0, min(5.0, value)), 2)
