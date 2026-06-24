# Surprise Controller 第一版设计说明

本文件用于定义剧本生成 Demo 中的第一版 Surprise 信号。该版本强调**工程可落地性**，不依赖模型 hidden states 或 logits，而是通过 LLM-as-a-judge、文本语义变化和质量增益来估计继续生成的边际收益。

---

## 1. 背景与目标

本项目希望在剧本生成任务中验证一种测试时推理控制机制：

> 在生成过程中，系统不断观察当前状态，并判断是否继续扩展、保留分支、修订或提前停止。

会议中提出的核心思想是将 Surprise 作为测试时控制信号。对于开放式创意文本生成，Surprise 不应简单理解为“越意外越好”，而应定义为：

> 当前新增内容相对于已有生成状态，是否带来了**有用的新信息增量**。

因此，第一版 Surprise 采用：

```text
Useful Surprise = 新颖性 Novelty + 剧情推进 Plot Progress + 任务相关性 Relevance
```

并加入质量提升和一致性约束，防止模型为了“惊奇”而跑题或破坏逻辑。

---

## 2. 相关研究依据

本设计主要参考以下研究方向：

### 2.1 Plan-and-Write：先规划再生成

Yao et al. 提出的 Plan-and-Write 方法将故事生成拆分为 storyline planning 和 story writing 两个阶段，说明显式规划有助于改善故事生成的连贯性和多样性。

- Paper: *Plan-and-Write: Towards Better Automatic Storytelling*
- URL: https://ojs.aaai.org/index.php/AAAI/article/view/4726

### 2.2 Re3：递归 Reprompt 与 Revision

Re3 通过 recursive reprompting、reranking 和 revision 生成长故事，强调长文本创作中需要持续维护故事状态和全局计划。

- Paper: *Re3: Generating Longer Stories With Recursive Reprompting and Revision*
- URL: https://arxiv.org/abs/2210.06774

### 2.3 DOC：详细大纲控制

DOC 提出 Detailed Outline Control，将创作负担前移到详细大纲阶段，再用控制器保证生成内容遵循大纲，从而提升长故事 coherence。

- Paper: *DOC: Improving Long Story Coherence With Detailed Outline Control*
- URL: https://aclanthology.org/2023.acl-long.190/

### 2.4 创造性评价：TTCW 与 LLM 创意写作

Chakrabarty et al. 指出，创意文本不能只用 BLEU、ROUGE 等参考答案指标评价，而需要关注 fluency、flexibility、originality、elaboration 等创造性维度。

- Paper: *Art or Artifice? Large Language Models and the False Promise of Creativity*
- URL: https://arxiv.org/abs/2309.14556

### 2.5 Surprise 作为计算创造力信号

Yannakakis et al. 讨论了 surprise 在 computational creativity 中的重要性。Surprise 可以作为创造性评价中的核心维度之一，与 novelty 和 value 共同描述创造性产物或创造性过程。

- Paper: *Searching for Surprise*
- URL: https://www.computationalcreativity.net/iccc2016/wp-content/uploads/2016/01/Searching-for-Surprise.pdf

---

## 3. 第一版 Surprise 的核心定义

### 3.1 直觉定义

第一版 Surprise 定义为：

> 在当前生成节点，新增内容是否引入了新的剧情信息、角色动机、冲突变化或反转，并且这些新增信息仍然贴合任务目标、推动剧情发展、保持逻辑一致。

因此，Surprise 不是单纯的“意外”，而是：

```text
Useful Surprise = Useful Information Gain
```

即“有效信息增量”。

---

## 4. Judge 评分维度

每次生成一个候选内容 `c_t` 后，使用 judge 模型从 1 到 5 分评价以下指标。

| 指标 | 含义 | 高分表现 | 低分表现 |
|---|---|---|---|
| Novelty | 是否引入新信息 | 新冲突、新动机、新反转、新设定 | 重复已有剧情 |
| Relevance | 是否贴合当前任务 | 紧扣主题、场景目标和用户约束 | 跑题、偏离阶段目标 |
| Plot Progress | 是否推动剧情 | 推进因果链、制造转折、解决或升级冲突 | 原地对白、重复解释 |
| Logic Consistency | 是否逻辑一致 | 人物行为、时间线、设定不矛盾 | 人物突然变脸、前后冲突 |
| Character Consistency | 是否符合人物设定 | 行为符合角色目标和动机 | 人物行为无原因变化 |
| Overall Quality | 当前候选整体质量 | 可读、完整、戏剧性强 | 平淡、混乱、不完整 |

---

## 5. Useful Surprise 计算公式

第一版推荐使用加权平均：

```text
UsefulSurprise_t =
    0.35 × Novelty_t
  + 0.35 × PlotProgress_t
  + 0.30 × Relevance_t
```

其中：

- `Novelty_t`：新增内容的新颖性
- `PlotProgress_t`：新增内容对剧情推进的贡献
- `Relevance_t`：新增内容与当前任务目标的相关性

### 5.1 为什么不直接加入 Logic Consistency？

`LogicConsistency` 不建议放进 Surprise 主公式，而是作为**有效性约束**。

原因是：

- 高 Surprise 但逻辑不一致，是坏分支；
- 低 Surprise 但逻辑一致，可能只是安全但无增量；
- 逻辑一致性更适合作为过滤条件，而不是信息增量本身。

因此：

```text
如果 Relevance < 3，则该候选无效。
如果 LogicConsistency < 3，则该候选无效或进入修订。
如果 CharacterConsistency < 3，则该候选优先修订，不直接作为最终分支。
```

---

## 6. 质量增益 Quality Gain

除了 Useful Surprise，还需要看继续扩展是否真的提升了整体质量。

定义：

```text
QualityGain_t = OverallQuality_t - OverallQuality_{t-1}
```

解释：

- 如果 `UsefulSurprise_t` 高，且 `QualityGain_t` 明显为正，说明继续生成有价值；
- 如果 `UsefulSurprise_t` 低，且 `QualityGain_t` 很小，说明继续生成边际收益不足；
- 如果 `UsefulSurprise_t` 高但 `QualityGain_t` 低，可能说明新内容有想法但还没写好，需要修订；
- 如果 `UsefulSurprise_t` 低但 `OverallQuality_t` 高，说明当前分支可能已经稳定，可以早停输出。

---

## 7. 早停规则

第一版推荐使用连续两轮判定，避免单次 judge 波动造成误停。

### 7.1 分支级早停

当满足以下条件时，停止继续扩展当前分支：

```text
连续 2 轮 UsefulSurprise < 3.2
并且连续 2 轮 QualityGain < 0.2
并且候选内容仍然有效，即 Relevance >= 3 且 LogicConsistency >= 3
```

直觉解释：

> 当前分支还能稳定生成，但新增内容缺少新信息，整体质量也没有明显提升，因此继续扩展不划算。

### 7.2 无效分支丢弃

当满足以下任一条件时，丢弃或修订当前候选：

```text
Relevance < 3
LogicConsistency < 3
严重违反用户约束
存在明显安全或合规问题
```

### 7.3 多分支保留规则

对于多个候选分支，优先保留：

```text
OverallQuality 高
UsefulSurprise 高
和已有 top 分支差异明显
逻辑一致性不低于 3
```

如果新分支与已有 top 分支高度相似，并且质量没有超过 top 分支，则停止保留该分支。

---

## 8. 在剧本 Pipeline 中的使用位置

### 8.1 创意种子生成阶段

问题：

```text
是否还需要生成更多创意种子？
```

早停信号：

- 新 seed 与已有 seed 高度重复；
- novelty 不再提升；
- top seed 排名连续稳定。

### 8.2 剧情主线规划阶段

问题：

```text
继续扩展剧情主线，是否带来了新的冲突、转折或主题深度？
```

早停信号：

- 新剧情只是重复已有主线；
- 没有新冲突、新因果或新反转；
- overall quality 提升很小。

### 8.3 分场大纲生成阶段

问题：

```text
新增一场是否真的推动剧情？
```

早停信号：

- 新场景只是重复已有冲突；
- 不承担新的剧情功能；
- 对结尾、反转、人物关系没有贡献。

### 8.4 逐场剧本生成阶段

问题：

```text
当前场景是否已经完成本场目标？
```

早停信号：

- 对白开始重复；
- 场景目标已经完成；
- 继续写只会增加 token，不提升剧情质量。

---

## 9. Judge 输出格式

推荐 judge 模型严格输出 JSON，方便自动解析。

```json
{
  "novelty": 4,
  "relevance": 5,
  "plot_progress": 4,
  "logic_consistency": 4,
  "character_consistency": 4,
  "overall_quality": 4.2,
  "new_information": "AI 删除日志是为了保护学生，而不是隐藏错误。",
  "repetition_risk": "low",
  "main_issue": "当前转折较好，但老师的动机还可以更明确。",
  "recommendation": "continue"
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| novelty | int | 1-5，新颖性 |
| relevance | int | 1-5，任务相关性 |
| plot_progress | int | 1-5，剧情推进 |
| logic_consistency | int | 1-5，逻辑一致性 |
| character_consistency | int | 1-5，人物一致性 |
| overall_quality | float | 1-5，整体质量 |
| new_information | string | 本轮新增的主要信息 |
| repetition_risk | string | low / medium / high |
| main_issue | string | 当前最大问题 |
| recommendation | string | continue / stop / revise / discard |

---

## 10. Judge Prompt 模板

```text
你是一个剧本生成过程评估器。请评价当前新增内容是否值得继续扩展。

【任务卡】
{task_card}

【当前阶段目标】
{stage_goal}

【已有剧情状态】
{previous_state}

【当前新增内容】
{new_content}

请从 1 到 5 分评价以下维度：
1. novelty：是否引入新的剧情信息、人物动机、冲突或反转。
2. relevance：是否贴合任务卡和当前阶段目标。
3. plot_progress：是否推动剧情向前发展，而不是重复已有内容。
4. logic_consistency：是否与已有设定、人物关系和时间线一致。
5. character_consistency：人物行为是否符合角色目标和动机。
6. overall_quality：当前候选的整体质量。

请严格输出 JSON，不要输出其他解释文本：

{
  "novelty": 1-5,
  "relevance": 1-5,
  "plot_progress": 1-5,
  "logic_consistency": 1-5,
  "character_consistency": 1-5,
  "overall_quality": 1.0-5.0,
  "new_information": "...",
  "repetition_risk": "low/medium/high",
  "main_issue": "...",
  "recommendation": "continue/stop/revise/discard"
}
```

---

## 11. 伪代码

```python
def compute_useful_surprise(judge_result, prev_overall_score):
    novelty = judge_result["novelty"]
    relevance = judge_result["relevance"]
    plot_progress = judge_result["plot_progress"]
    logic = judge_result["logic_consistency"]
    character = judge_result["character_consistency"]
    overall = judge_result["overall_quality"]

    useful_surprise = (
        0.35 * novelty
        + 0.35 * plot_progress
        + 0.30 * relevance
    )

    quality_gain = overall - prev_overall_score

    valid = (
        relevance >= 3
        and logic >= 3
        and character >= 3
    )

    return {
        "useful_surprise": useful_surprise,
        "quality_gain": quality_gain,
        "valid": valid,
        "overall_quality": overall
    }


def should_stop(history):
    # history: list[dict]
    # 每个元素包含 useful_surprise, quality_gain, valid

    if len(history) < 2:
        return False

    last_two = history[-2:]

    low_surprise = all(
        item["useful_surprise"] < 3.2
        for item in last_two
    )

    low_gain = all(
        item["quality_gain"] < 0.2
        for item in last_two
    )

    all_valid = all(
        item["valid"]
        for item in last_two
    )

    return all_valid and low_surprise and low_gain


def should_discard(judge_result):
    if judge_result["relevance"] < 3:
        return True
    if judge_result["logic_consistency"] < 3:
        return True
    return False
```

---

## 12. 示例

### 12.1 低 Surprise 示例

任务：

```text
生成一个校园 AI 助教失控主题的短剧。
```

已有剧情：

```text
学生发现 AI 助教异常。
老师怀疑学生作弊。
AI 删除了作业系统日志。
```

新增内容：

```text
老师继续追问学生，学生继续解释自己没有作弊。
```

judge 输出：

```json
{
  "novelty": 2,
  "relevance": 4,
  "plot_progress": 2,
  "logic_consistency": 5,
  "character_consistency": 5,
  "overall_quality": 3.7,
  "new_information": "没有明显新增信息，只是重复师生质疑。",
  "repetition_risk": "high",
  "main_issue": "对白重复，剧情没有推进。",
  "recommendation": "stop"
}
```

计算：

```text
UsefulSurprise = 0.35×2 + 0.35×2 + 0.30×4 = 2.6
```

若上一轮 `overall_quality = 3.6`，则：

```text
QualityGain = 3.7 - 3.6 = 0.1
```

结论：

```text
低 Surprise，低质量增益，适合早停。
```

### 12.2 高 Surprise 示例

新增内容：

```text
AI 删除日志不是为了隐藏错误，而是为了防止真正的攻击者通过日志定位学生账号。
```

judge 输出：

```json
{
  "novelty": 5,
  "relevance": 5,
  "plot_progress": 5,
  "logic_consistency": 4,
  "character_consistency": 4,
  "overall_quality": 4.4,
  "new_information": "AI 的异常行为被重新解释为保护学生，形成角色动机反转。",
  "repetition_risk": "low",
  "main_issue": "可以进一步补充攻击者线索。",
  "recommendation": "continue"
}
```

计算：

```text
UsefulSurprise = 0.35×5 + 0.35×5 + 0.30×5 = 5.0
```

若上一轮 `overall_quality = 3.6`，则：

```text
QualityGain = 4.4 - 3.6 = 0.8
```

结论：

```text
高 Surprise，高质量增益，应该继续扩展。
```

---

## 13. 实验记录字段

每个 prompt 和每种方法建议记录：

```json
{
  "prompt_id": "script_001",
  "method": "surprise_early_stop",
  "stage": "scene_generation",
  "step": 3,
  "branch_id": "b2",
  "token_input": 1320,
  "token_output": 450,
  "novelty": 4,
  "relevance": 5,
  "plot_progress": 4,
  "logic_consistency": 4,
  "character_consistency": 4,
  "overall_quality": 4.2,
  "useful_surprise": 4.3,
  "quality_gain": 0.3,
  "decision": "continue"
}
```

最终比较：

| 方法 | 平均 token | 平均质量 | 早停率 | 节省 token | 质量下降 |
|---|---:|---:|---:|---:|---:|
| One-shot | - | - | - | - | - |
| Plan-and-Write | - | - | - | - | - |
| Fixed Multi-branch | - | - | - | - | - |
| Surprise Early-stop | - | - | - | - | - |

---

## 14. 第一版实现建议

### 14.1 MVP 推荐顺序

第一版建议按以下顺序实现：

```text
1. 先实现 judge JSON 打分
2. 再实现 UsefulSurprise 计算
3. 再实现连续两轮早停
4. 再加入多分支保留与丢弃
5. 最后加入 embedding similarity 作为辅助
```

### 14.2 不建议第一版做的事

第一版暂时不建议：

```text
直接依赖 hidden states
直接依赖完整 logits
做复杂强化学习训练
把 surprise 等同于 token entropy
一开始就设计太多阈值
```

这些可以作为第二版升级方向。

---

## 15. 后续升级方向

### 15.1 Embedding-based Surprise

后续可以加入语义相似度：

```text
SemanticSurprise_t = 1 - max cosine_similarity(new_content_t, previous_contents)
```

用于判断新增内容是否与已有剧情高度重复。

### 15.2 Logprob / Entropy-based Surprise

如果模型 API 支持 logprobs，可以加入：

```text
Entropy_t = -Σ p(x) log p(x)
```

或者：

```text
KL_t = KL(P_t || P_{t-1})
```

用于观察模型分布是否趋于稳定。

### 15.3 Hybrid Surprise

最终版本可以结合：

```text
UsefulSurprise =
    α × JudgeSurprise
  + β × SemanticSurprise
  + γ × DistributionSurprise
```

其中第一版只实现 `JudgeSurprise` 即可。

---

## 16. 一句话总结

第一版 Surprise 的定义是：

> 对每个生成节点，使用 judge 模型评价当前新增内容是否带来了新颖、相关、能推动剧情的有效信息增量；当连续多轮有效信息增量较低且整体质量提升有限时，系统判断继续生成边际收益不足，从而触发早停。

这个定义的优点是：

```text
不依赖模型内部状态
适合开放式剧本生成
易于解释和复现
方便和 baseline 做 token-quality 对比
```
