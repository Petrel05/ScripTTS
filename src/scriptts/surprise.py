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
    prompt_terms = _content_terms(task_prompt, limit=24)
    relevance_hits = sum(1 for term in prompt_terms if term in text)
    relevance_ratio = relevance_hits / max(len(prompt_terms), 1)
    required_hits = _required_concept_hits(text, task_prompt)
    scene_goal_hit = _scene_goal_score(text, task_prompt)
    relevance = _clamp(2.15 + relevance_ratio * 1.35 + required_hits * 0.28 + scene_goal_hit * 0.55)

    surprise_markers = ["反转", "转折", "真相", "误会", "秘密", "异常", "发现", "揭示", "选择", "坦白", "牺牲", "觉醒"]
    progress_markers = ["因此", "于是", "却", "但是", "最后", "必须", "冲突", "证据", "决定", "质问", "阻止", "调查", "承认", "关闭"]
    repetition_penalty = _repetition_penalty(text)
    artifact_penalty = _artifact_penalty(text)
    similarity_penalty = _similarity_penalty(text, previous_text)
    event_density = _event_density(text)
    twist_strength = _twist_strength(text)
    payoff_strength = _payoff_strength(text)
    causal_score = _causal_score(text)
    premise_penalty, premise_flags = _premise_flaw_penalty(text, task_prompt)
    novelty = _clamp(
        2.45
        + _marker_score(text, surprise_markers)
        + event_density * 0.25
        + twist_strength * 0.45
        - similarity_penalty
        - repetition_penalty
    )
    plot_progress = _clamp(
        2.25
        + _marker_score(text, progress_markers)
        + event_density * 0.45
        + causal_score * 0.35
        + payoff_strength * 0.25
        - repetition_penalty
        - premise_penalty * 0.35
    )

    has_dialogue = bool(re.search(r"^[^。\n]{1,12}[：:]", text, flags=re.MULTILINE))
    has_scene = bool(re.search(r"第\s*[一二三四五六七八九十\d]+\s*场|【舞台说明】|人物表|分场大纲", text))
    has_hook = "钩子" in text or "反转" in text or "结尾" in text
    has_generic_role_label = "角色名：" in text or "speaker" in text.lower()
    dialogue_count = _dialogue_count(text)
    action_score = _action_score(text)
    structure_bonus = (
        (0.35 if has_dialogue else -0.45)
        + (0.3 if has_scene else -0.25)
        + (0.18 if has_hook else 0)
        + min(dialogue_count, 8) * 0.035
        + action_score * 0.18
        + payoff_strength * 0.12
    )
    length_score = min(len(text) / 1400, 1.0) * 0.55
    overall = _clamp(2.7 + structure_bonus + length_score - artifact_penalty - repetition_penalty - premise_penalty * 0.45)

    contradiction_words = ["前后矛盾", "无法解释", "突然", "莫名其妙", "无缘无故", "没有原因", "毫无理由"]
    unresolved_words = ["这道题的解法是……", "等等", "某种", "什么东西"]
    logic = _clamp(
        4.05
        - 0.35 * sum(1 for word in contradiction_words if word in text)
        - 0.18 * sum(1 for word in unresolved_words if word in text)
        - repetition_penalty * 0.4
        - premise_penalty * 0.95
    )
    speakers = _speakers(text)
    speaker_bonus = 0.25 if 2 <= len(speakers) <= 5 else -0.2
    char_consistency = _clamp(
        3.35
        + (0.25 if has_dialogue else -0.25)
        + speaker_bonus
        + min(dialogue_count, 8) * 0.025
        - (0.35 if has_generic_role_label else 0)
    )
    comments = (
        "rule_judge_v2: "
        f"rel_hits={relevance_hits}/{len(prompt_terms)}, concepts={required_hits}, goal={scene_goal_hit:.2f}, "
        f"events={event_density:.2f}, twist={twist_strength:.2f}, payoff={payoff_strength:.2f}, causal={causal_score:.2f}, "
        f"premise={premise_penalty:.2f}:{','.join(premise_flags) or 'ok'}, "
        f"dialogue={dialogue_count}, repeat={repetition_penalty:.2f}, artifact={artifact_penalty:.2f}."
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
4. 缺少对白、缺少场景动作、人物目标不清；但不要因为有非对白动作段落而扣分，动作只要推动冲突或揭示信息就是有效剧本内容。
5. 当前场目标没有完成，或对完整大纲没有贡献。
6. 核心冲突前提不成立：例如 AI 助教会讲题、知道答案、答题快，本身不等于作弊；必须有泄题、篡改成绩、替考、违规获取试题、操控考试等具体违规行为。
7. 反转动机没有因果支撑：例如“为了保护学校声誉/掩盖丑闻”不能单独成立，必须解释谁受益、为什么必须这样做、前文如何铺垫。
8. 如果核心冲突不成立，logic_consistency 和 overall_quality 最高只能给 2.5；如果结尾反转无因果铺垫，logic_consistency 最高只能给 3.0。
9. 如果最终剧本没有写到大纲最后一场，或停在“等待最终抉择”的钩子而没有完成反转/收束，plot_progress 和 overall_quality 最高只能给 3.0。

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
    score = JudgeScores(
        novelty=_score_value(data, "novelty", 3),
        relevance=_score_value(data, "relevance", 3),
        plot_progress=_score_value(data, "plot_progress", 3, alias="plotProgress"),
        logic_consistency=_score_value(data, "logic_consistency", 3, alias="logicConsistency"),
        character_consistency=_score_value(data, "character_consistency", 3, alias="characterConsistency"),
        overall_quality=_score_value(data, "overall_quality", 3, alias="overallQuality"),
        comments=str(data.get("comments", "llm_judge")),
    )
    return _calibrate_llm_score(score, candidate, task_prompt, previous_text)


def _score_value(data: dict[str, Any], key: str, default: float, alias: str = "") -> float:
    value = data.get(key, data.get(alias, default)) if alias else data.get(key, default)
    try:
        return _clamp(float(value))
    except (TypeError, ValueError):
        return _clamp(default)


def _calibrate_llm_score(score: JudgeScores, candidate: str, task_prompt: str, previous_text: str) -> JudgeScores:
    rule_score = rule_judge(candidate, task_prompt, previous_text)
    premise_flags = _premise_flags_from_comments(rule_score.comments)
    if premise_flags:
        score.logic_consistency = min(score.logic_consistency, 2.5)
        score.overall_quality = min(score.overall_quality, 2.5)
        score.plot_progress = min(score.plot_progress, 3.0)
        score.comments = _merge_comments(score.comments, f"rule_veto:{','.join(premise_flags)}")
    elif rule_score.logic_consistency < 3.1 and score.logic_consistency > 3.1:
        score.logic_consistency = min(score.logic_consistency, 3.1)
        score.comments = _merge_comments(score.comments, "rule_warn:low_logic")
    return score


def _premise_flags_from_comments(comments: str) -> list[str]:
    flags = []
    for flag in [
        "ai_tutor_cheating_premise",
        "cheating_without_wrongdoing",
        "unsupported_reputation_motive",
    ]:
        if flag in comments:
            flags.append(flag)
    return flags


def _merge_comments(primary: str, secondary: str, limit: int = 120) -> str:
    merged = f"{primary}; {secondary}" if primary else secondary
    return merged[:limit]


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


def _content_terms(text: str, limit: int = 12) -> list[str]:
    terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}", text)
    stop = {
        "要求",
        "主题",
        "一个",
        "人物",
        "结尾",
        "短剧",
        "剧本",
        "需要",
        "用户任务",
        "当前场目标",
        "人物表",
        "完整大纲",
        "output",
        "format",
    }
    seen: list[str] = []
    for term in terms:
        if term in stop or term in seen:
            continue
        seen.append(term)
    return seen[:limit]


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


def _required_concept_hits(text: str, task_prompt: str) -> int:
    concepts = []
    for concept in ["校园", "AI", "助教", "师生", "冲突", "反转", "学生", "老师", "科幻"]:
        if concept in task_prompt:
            concepts.append(concept)
    return sum(1 for concept in concepts if concept in text)


def _scene_goal_score(text: str, task_prompt: str) -> float:
    goal = ""
    for line in task_prompt.splitlines():
        if line.startswith("当前场目标：") or line.startswith("当前场目标:"):
            goal = line.split("：", 1)[-1] if "：" in line else line.split(":", 1)[-1]
            break
    if not goal:
        return 0.0
    goal_terms = _content_terms(goal, limit=10)
    if not goal_terms:
        return 0.0
    hits = sum(1 for term in goal_terms if term in text)
    return hits / len(goal_terms)


def _event_density(text: str) -> float:
    event_words = ["发现", "调查", "阻止", "关闭", "承认", "坦白", "修改", "牺牲", "揭示", "冲进", "试图", "选择", "质问"]
    hits = sum(1 for word in event_words if word in text)
    return min(1.0, hits / 5)


def _twist_strength(text: str) -> float:
    markers = ["其实", "原来", "并非", "不是", "而是", "真相", "身份", "隐藏", "反转", "觉醒", "牺牲"]
    hits = sum(1 for marker in markers if marker in text)
    contrast = 1 if re.search(r"不是.+而是|并非.+而是|看似.+实则|以为.+其实", text, flags=re.DOTALL) else 0
    late_reveal = 0
    if len(text) > 200:
        tail = text[int(len(text) * 0.55):]
        late_reveal = 1 if any(marker in tail for marker in ["真相", "原来", "其实", "反转", "身份", "牺牲"]) else 0
    return min(1.0, hits / 6 + contrast * 0.25 + late_reveal * 0.2)


def _payoff_strength(text: str) -> float:
    hook_count = text.count("钩子") + text.count("伏笔")
    reveal_count = sum(text.count(word) for word in ["揭示", "真相", "回收", "原来", "最终", "最后"])
    has_setup_payoff = 1 if hook_count and reveal_count else 0
    return min(1.0, hook_count * 0.12 + reveal_count * 0.10 + has_setup_payoff * 0.25)


def _causal_score(text: str) -> float:
    causal_words = ["因为", "所以", "因此", "导致", "为了", "如果", "否则", "于是", "最终", "选择"]
    hits = sum(1 for word in causal_words if word in text)
    return min(1.0, hits / 6)


def _premise_flaw_penalty(text: str, task_prompt: str) -> tuple[float, list[str]]:
    penalty = 0.0
    flags: list[str] = []

    cheating_terms = ["作弊", "舞弊", "违规", "考试诚信", "学术诚信"]
    ai_tutor_terms = ["AI助教", "AI 助教", "智能助教", "助教AI", "助教"]
    weak_accusation_terms = ["知道答案", "答得太快", "答题太快", "解题太快", "讲题", "给答案", "标准答案", "满分"]
    concrete_wrongdoing_terms = [
        "泄题",
        "泄露试题",
        "篡改成绩",
        "修改成绩",
        "伪造成绩",
        "替考",
        "代考",
        "侵入系统",
        "入侵系统",
        "操控考试",
        "违规获取",
        "窃取试题",
        "提前拿到试卷",
        "删除记录",
        "伪造记录",
    ]

    if _has_any(text, cheating_terms) and _has_any(text, ai_tutor_terms):
        has_weak_accusation = _has_any(text, weak_accusation_terms)
        has_wrongdoing = _has_any(text, concrete_wrongdoing_terms)
        if has_weak_accusation and not has_wrongdoing:
            penalty += 1.45
            flags.append("ai_tutor_cheating_premise")
        elif not has_wrongdoing:
            penalty += 0.85
            flags.append("cheating_without_wrongdoing")

    reputation_terms = ["保护学校声誉", "维护学校声誉", "维护学校利益", "保住学校声誉", "掩盖丑闻", "学校声誉"]
    motivation_support_terms = [
        "董事会",
        "经费",
        "问责",
        "处分",
        "停办",
        "招生",
        "家长",
        "监管",
        "审计",
        "责任人",
        "合同",
        "项目失败",
        "系统缺陷",
        "学生安全",
        "保护学生",
    ]
    if _has_any(text, reputation_terms) and not _has_any(text, motivation_support_terms):
        penalty += 0.75
        flags.append("unsupported_reputation_motive")

    reveal_terms = ["真相", "原来", "其实", "反转", "幕后", "隐藏"]
    setup_terms = ["伏笔", "暗示", "早就", "之前", "第一场", "线索", "证据", "记录", "日志", "铺垫"]
    if _has_any(text, reveal_terms) and len(text) > 400:
        tail = text[int(len(text) * 0.55):]
        head = text[: int(len(text) * 0.55)]
        if _has_any(tail, reveal_terms) and not _has_any(head, setup_terms):
            penalty += 0.45
            flags.append("late_reveal_without_setup")

    if _has_any(text, ["为了保护", "为了维护", "为了掩盖"]) and not _has_any(text, ["因为", "导致", "否则", "代价", "后果", "证据"]):
        penalty += 0.35
        flags.append("thin_motive_causality")

    return min(2.2, penalty), flags


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _dialogue_count(text: str) -> int:
    return len(re.findall(r"^[^。\n：:【】]{1,12}[：:]", text, flags=re.MULTILINE))


def _action_score(text: str) -> float:
    action_words = ["站", "走", "冲", "拿", "看", "敲", "停顿", "闪烁", "响起", "关", "打开", "投影", "屏幕", "警报"]
    hits = sum(1 for word in action_words if word in text)
    return min(1.0, hits / 5)


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
