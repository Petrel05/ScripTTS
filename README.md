# ScripTTS — Surprise-Controlled Script Generation Pipeline

> 将剧本生成建模为**可观察、可控制、可早停的测试时推理过程**。  
> 核心思想：用 Useful Surprise 作为动态控制信号，在 LLM 生成过程中实时决策继续、分叉、剪枝或早停，从而在保持质量的同时减少 token 消耗。

---

## 1. 一句话总结

**评分驱动的实时树搜索**：LLM 每输出一段内容就立即被评判（6 维度 + Useful Surprise），分数直接决定该分支/该场是继续展开、被剪掉还是提前收束——不是事后评估，而是用评分来动态控制生成过程本身。

---

## 2. Pipeline 架构

```
User Prompt
  → Stage 0  需求规范化 (task_card)
  → Stage 1  创意种子生成 (seed) ×N → judge → prune
  → Stage 2  剧情主线规划 (storyline)   → judge → fork / prune
  → Stage 3  人物与冲突设计 (characters) → judge → prune
  → Stage 4  分场大纲 (outline)         → judge → prune
  → Stage 5  逐场剧本生成 (scenes)      → 每场 judge → stop / prune
  → Final Judge → 选最优分支 → 输出剧本 + trace
```

每个 `生成 → judge → 决策` 循环都是**紧耦合的**，controller 在每个阶段之后都会运行剪枝逻辑，而不是等所有内容生成完再回头挑选。

---

## 3. 分数计算原理

### 3.1 六个评分维度 (1–5 分)

| 维度 | 含义 | 高分表现 |
|------|------|---------|
| `novelty` | 新颖性 | 引入新冲突、新动机、新反转 |
| `relevance` | 任务相关性 | 紧扣主题和当前阶段目标 |
| `plot_progress` | 剧情推进度 | 推动因果链、制造转折 |
| `logic_consistency` | 逻辑一致性 | 人物行为、时间线不矛盾 |
| `character_consistency` | 角色一致性 | 行为符合角色目标和动机 |
| `overall_quality` | 整体质量 | 可读、完整、戏剧性强 |

### 3.2 核心指标：Useful Surprise

```python
useful_surprise = 0.35 × novelty + 0.35 × plot_progress + 0.30 × relevance
```

定义为**有效信息增量**——不是越意外越好，而是新内容是否推动剧情 + 是否新颖 + 是否相关。`logic_consistency` 不进入公式，而是作为硬门槛。

### 3.3 分支排名 `_branch_rank`

```python
base = overall_quality × 0.35
     + useful_surprise  × 0.25
     + logic_consistency × 0.28
     + character_consistency × 0.12

# 硬惩罚
if logic_consistency < 3.1:   base -= 1.2 + (3.1 - logic) × 0.7
if premise_flaw:              base -= 1.0

rank = base + depth_bonus + diversity_bonus + gain_bonus + final_bonus
```

**逻辑一致性和前提缺陷是最严厉的扣分项**。

### 3.4 两种评判后端

| 后端 | 文件 | 原理 |
|------|------|------|
| `rule` | [surprise.py](src/scriptts/surprise.py#L31) | ~400 行规则引擎，关键词/句式/结构/因果链特征工程 |
| `llm` | [surprise.py](src/scriptts/surprise.py#L122) | LLM 生成 JSON 评分 (temperature=0)，再经 rule 校准 veto |

**LLM + Rule Veto 组合**：LLM 先打分，rule 对硬伤（如前提缺陷）做覆盖——LLM 可能觉得"AI 保护学生"合理，rule 会检查是否有前文因果支撑。

---

## 4. Controller：何时触发 Action

Controller 在**生成过程中实时**做出以下决策：

### 4.1 分支无效判定 (`_invalid_reason`)

| 条件 | Action |
|------|--------|
| `relevance < 2.5` | prune |
| 前提缺陷 (如 AI 知道答案 ≠ 作弊) | prune |
| `logic_consistency < 3.1` | prune |
| `overall_quality < 2.4` | prune |

### 4.2 阶段剪枝 (`_controller_prune`)

每次 stage 完成后执行三种剪枝：

1. **Active Prune Margin (默认 0.75)**：与最佳分支排名差距过大 → 剪掉
2. **Similarity Prune (默认 0.72)**：Jaccard 相似度过高 → 去重保留一个
3. **Width Limit (默认 2)**：最多保留 `max_active_branches` 个活跃分支

### 4.3 场景早停 (`should_stop`)

```python
if 连续两场 useful_surprise < 3.2
   and quality_gain < 0.2
   and 候选仍有效 (relevance≥3, logic≥3):
    → 提前停止该分支，不继续生成后续场
```

**这是省 token 的核心机制**：质量够了就停，不写满大纲计划的所有场。

### 4.4 Fork（分叉）

高分分支 (`rank ≥ fork_score_threshold=3.85`) 触发 fork：LLM 基于该分支生成一个"改变冲突或反转机制"的新兄弟分支。

### 4.5 Rescue & Fallback（容错）

- **Rescue**：如果所有初始种子都被 prune，复活最优种子继续
- **Fallback Scenes**：如果没有任何 scene 生成成功，从 outline 合成降级场景
- **Fallback Final**：如果没有合格候选，选最好的"坏结果"并在 trace 中标记

### 4.6 Final Judge 补全校准

最终评判会检查：
- 大纲计划 N 场，实际生成 M 场 (M < N) → `plot_progress` 和 `overall_quality` 强制 ≤ 3.0
- 最后一场停在钩子（"等待抉择"等）而非真正收束 → 强制压低分数

---

## 5. Token 消耗模型与控制收益

Controller 的 prune / stop 决策会减少后续调用，但 judge 本身也是开销。本节给出精确的成本分解和节省来源。

### 5.1 调用次数上限（无任何干预时）

默认配置 `min_branches=3, max_branches=2, max_active_branches=2, max_scenes=4` 下，若 controller 完全不干预（不剪枝、不早停、不 fork），调用次数为：

| 阶段 | Gen 调用 | Judge 调用 | 说明 |
|------|---------|-----------|------|
| Stage 0 规范化 | 1 | 0 | |
| Stage 1 Seed ×2 | 2 | 2 | `initial_count = min(max(3,1), 2) = 2` |
| Fork | 0 | 0 | `branch_count(2) ≥ max_branches(2)`，不触发 |
| Stage 2 Storyline ×2 | 2 | 2 | |
| Stage 3 Characters ×2 | 2 | 2 | |
| Stage 4 Outline ×2 | 2 | 2 | |
| Stage 5 Scenes ×(2×4) | 8 | 8 | |
| Final Judge ×2 | 0 | 2 | judge only |
| **合计** | **17** | **18** | **35 次** |

> 注：Gen 调用 `max_new_tokens` 默认 768（可能触发 retry 翻倍至 1536 甚至 3072），Judge 调用默认 512。上表不含 retry，每次 retry 额外 +1 gen 调用。

### 5.2 每次干预节省的调用数

以 35 次为无干预上限，每次剪枝/早停实际节省的调用数取决于发生时机：

| 干预时机 | 已消耗（该分支） | 被跳过的阶段 | 节省调用 |
|----------|-----------------|-------------|---------|
| Seed 后 prune | gen+judge=2 | Storyline + Char + Outline + 4 Scenes + Final | **15** |
| Storyline 后 prune | 4 | Char + Outline + 4 Scenes + Final | **13** |
| Characters 后 prune | 6 | Outline + 4 Scenes + Final | **11** |
| Outline 后 prune | 8 | 4 Scenes + Final | **9** |
| Scene 2 后早停 | 已完成 2 场 | 剩余 2 场的 gen+judge | **4** |
| Similarity dedup | 视去重阶段而定 | 同对应阶段的 prune | 9–15 |

**逻辑**：越早剪掉劣质分支，节省越多。Seed 阶段一次 prune（15 调用）约等于节省了 43% 的最大调用预算。

### 5.3 Judge 开销是否值得

相比"不用 controller，所有分支跑完再评选"的方案（17 gen + 2 final judge = 19 调用），controller 额外增加了 16 次 intermediate judge（35 − 19）。

- **若 1 个分支在 Stage 2 被 prune**：额外开销 16 judge，节省 13 调用 → 净增 3 调用，但避免了 4 场低质量 scene 的生成。
- **若 1 个分支在 Stage 1 被 prune**：额外 16，节省 15 → 接近持平。
- **若 0 个分支被 prune**：controller 是纯开销（+16 judge），但这只有在两个分支质量完全相等且都很好的理想情况下才会发生。

**结论**：controller 的净收益来自**避免在劣质分支上浪费昂贵的 scene 生成调用**（每个 scene gen 输出上限 768 token，而 judge 输出上限 512 token）。只要至少剪掉 1 个分支，token 层面基本持平或略省；真正的收益是质量——避免把劣质内容写进最终剧本。

### 5.4 何处真正省 Token

Controller 省的不是 judge 调用本身，而是**被剪分支后续 stage 的 gen 调用**。Gen 调用的 prompt 随上下文累积而膨胀（scene 阶段 prompt 包含 task_card + storyline + characters + outline + previous_text，可达 1500+ token 输入），输出上限 768 token。Judge 调用的 prompt 较短（candidate 截断 + task_prompt），输出上限 512 token。因此：

- 每跳过一次 scene gen → 节省 ~1500 input + ~768 output token
- 每跳过一次 judge → 节省 ~800 input + ~200 output token
- 总账：跳过一个 gen 的收益大约是跳过一个 judge 的 2–3 倍

Controller 的设计目标就是**用便宜的 judge 调用去避免昂贵的 gen 调用**。

### 5.5 实验观测

| 实验 | 调用数 | vs 上限 35 | 说明 |
|------|--------|-----------|------|
| Qwen3-8B + LLM judge v1 | 34 | −1（~3%） | 几乎无剪枝 |
| Qwen3-8B + LLM judge v2 | 30 | −5（~14%） | 少量剪枝或早停 |
| DeepSeek v4 Pro v2 | 20 | −15（~43%） | 明显剪枝 + 早停 |

这些数字与模型行为直接相关：DeepSeek 生成质量更高、分支差异更大，controller 的相似度去重和 active-prune-margin 更容易触发；而本地 8B 模型两个分支常趋同，剪枝机会少。**节省率因模型而异，不应被当作常数。**

### 5.6 限制与注意事项

1. **没有实现无 controller 对照组**：目前 pipeline 始终运行 controller，上述节省估算是对"如果所有分支跑满所有阶段"的理论上限推算，而非实测对照。
2. **Retry 机制增加方差**：`finish_reason=length` 或 JSON 解析失败会触发重试（token limit 翻倍），实际调用数可能超出 35。真实情况下需要将 retry 成本也纳入。
3. **与 one-shot 的对比是不同维度**：one-shot 只需 1 次调用，但完全没有多分支探索和质量保障。controller 的价值不在与 one-shot 比 token 效率，而在于：相比 naive multi-branch（全跑完），controller 削减了不必要的开销；相比 one-shot，controller 用合理的额外预算换取了多分支探索带来的质量提升。
4. **节省率需要通过对照实验验证**：要得到可靠的 "token 节省 X%" 结论，需要在同一 prompt 集合上跑 "无 controller 的固定 multi-branch" 和 "带 controller 的动态多分支" 两组的对比实验。

---

## 6. 快速开始

### 本地 Smoke Test (无需模型)

```bash
python3 run_pipeline.py --backend mock --limit 1 --run-name smoke_mock
```

输出：
```
outputs/smoke_mock/results.jsonl
outputs/smoke_mock/scripts/script_001.md
outputs/smoke_mock/traces/script_001.md
```

### 本地 HF 模型

```bash
# 安装依赖
pip install torch transformers accelerate

# 最小模型单条
python3 run_pipeline.py \
  --backend hf \
  --model-path /data/fhy/models/Qwen3-0.6B \
  --prompt-id script_001 \
  --run-name qwen06b_one

# 8B 模型 + LLM judge
python3 run_pipeline.py \
  --backend hf \
  --model-path /data/fhy/models/Qwen3-8B \
  --judge-backend llm \
  --min-branches 3 --max-branches 5 \
  --max-active-branches 2 \
  --min-scenes 3 --max-scenes 4 \
  --max-new-tokens 1024 --scene-max-new-tokens 1536 \
  --temperature 0.55 \
  --run-name qwen8b_demo
```

### DeepSeek API

```bash
export DEEPSEEK_API_KEY="YOUR_KEY"

python3 run_pipeline.py \
  --backend deepseek \
  --api-model deepseek-v4-pro \
  --api-thinking disabled \
  --api-reasoning-effort medium \
  --prompt-id script_001 \
  --run-name deepseek_demo \
  --min-branches 3 --max-branches 5 \
  --max-active-branches 2 \
  --min-scenes 3 --max-scenes 4 \
  --max-new-tokens 1024 --scene-max-new-tokens 1536 \
  --temperature 0.55
```

### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--backend` | `auto` | `mock` / `hf` / `deepseek` |
| `--judge-backend` | `llm` | `rule` / `llm` / `hybrid` |
| `--max-branches` | 2 | 最大分支数 |
| `--min-branches` | 3 | 初始种子数 |
| `--max-active-branches` | 2 | 并行活跃分支上限 |
| `--max-scenes` | 4 | 最大场数 |
| `--min-scenes` | 3 | 最少场数 |
| `--max-new-tokens` | 768 | 普通阶段生成长度 |
| `--scene-max-new-tokens` | 768 | 单场生成长度 |
| `--temperature` | 0.7 | 采样温度 |
| `--fork-score-threshold` | 3.85 | 触发 fork 的排名阈值 |
| `--active-prune-margin` | 0.75 | 弱分支剪枝边距 |
| `--similarity-prune-threshold` | 0.72 | 相似分支去重阈值 |

---

## 7. 输出结构

```
outputs/{run_name}/
├── results.jsonl      # 完整结构化记录
├── scripts/
│   └── script_001.md  # 可读最终剧本
└── traces/
    └── script_001.md  # 完整决策链回放
```

`results.jsonl` 每行包含：

| 字段 | 内容 |
|------|------|
| `task` | 原始任务 |
| `task_card` | 规范化任务卡 |
| `branches` | 所有分支的状态/评分/决策 |
| `controller_events` | Controller 每次动作记录 |
| `chosen_seed` / `storyline` / `characters` / `outline` | 选中分支的各阶段 |
| `scenes` | 逐场剧本及每场评分 |
| `final_score` | 6 维度 + useful_surprise |
| `metrics` | API 调用次数、token、分支数、耗时 |

---

## 8. 近期实验

| 配置 | API Calls | Tokens | Overall | Surprise |
|------|-----------|--------|---------|----------|
| Qwen3-8B + LLM judge v1 | 34 | 35,109 | 3.8 | 3.825 |
| Qwen3-8B + LLM judge v2 | 30 | 32,565 | 3.2 | 4.045 |
| DeepSeek v4 Pro v2 | 20 | 24,803 | 2.0 | — |

观察：
- LLM judge 比 rule judge 更能避免"AI 知道答案 = 作弊"类前提错误
- DeepSeek 在 seed/storyline/characters 阶段质量显著高于本地 8B，但 JSON 截断问题需要 retry 机制
- 4 场完整写完但结尾停留在"等待抉择"仍是主要质量问题

---

## 9. 代码结构

```
ScripTTS/
├── run_pipeline.py              # CLI 入口
├── prompts.jsonl                # 输入任务
├── src/scriptts/
│   ├── pipeline.py              # 核心 Pipeline：分支管理、剪枝、fork、stop
│   ├── surprise.py              # 评分引擎：rule judge + LLM judge + should_stop
│   ├── llm.py                   # LLM 抽象层：Mock / OpenAI API / HF Local
│   ├── data.py                  # ScriptTask + JSONL 读写
│   └── cli.py                   # 参数解析、后端选择、配置组装
└── outputs/                     # 实验结果
```

### 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `ScriptPipeline.run_task` | [pipeline.py:145](src/scriptts/pipeline.py#L145) | 主流程编排 |
| `_judge` | [pipeline.py:462](src/scriptts/pipeline.py#L462) | 评分调度 (rule/llm) |
| `_controller_prune` | [pipeline.py:481](src/scriptts/pipeline.py#L481) | 三策略剪枝 |
| `_branch_rank` | [pipeline.py:545](src/scriptts/pipeline.py#L545) | 分支排名公式 |
| `_branch_scene_stop` | [pipeline.py:622](src/scriptts/pipeline.py#L622) | 场景级停止判断 |
| `_invalid_reason` | [pipeline.py:1199](src/scriptts/pipeline.py#L1199) | 分支无效判定 |
| `rule_judge` | [surprise.py:31](src/scriptts/surprise.py#L31) | 规则评分引擎 |
| `build_llm_judge_prompt` | [surprise.py:128](src/scriptts/surprise.py#L128) | LLM 评判 prompt |
| `should_stop` | [surprise.py:216](src/scriptts/surprise.py#L216) | 通用早停逻辑 |
| `_premise_flaw_penalty` | [surprise.py:338](src/scriptts/surprise.py#L338) | 前提缺陷检测 |
| `JudgeScores.useful_surprise` | [surprise.py:22](src/scriptts/surprise.py#L22) | Useful Surprise 公式 |

---

## 10. 相关论文

- **Plan-and-Write** (Yao et al., 2019) — 先规划 storyline 再生成
- **Re3** (Yang et al., 2022) — 递归 reprompting + revision
- **DOC** (Yang et al., 2023) — 详细大纲控制长故事 coherence
- **TTCW** (Chakrabarty et al., 2024) — 创意写作评价维度
- **Searching for Surprise** (Yannakakis et al., 2016) — Surprise 作为计算创造力信号
- **AutoTTS** (Zheng et al., 2026) — 测试时推理控制策略自动发现
- **Parallel-Probe** (Zheng et al., 2026) — 分支管理 + 共识早停
