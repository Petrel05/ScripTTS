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
  → Stage 1  创意种子生成 (seed) ×3 → judge → prune
  → Stage 2  剧情主线规划 (storyline)   → judge → fork / prune
  → Stage 3  人物与冲突设计 (characters) → judge → prune
  → Stage 4  分场大纲 (outline)         → judge → prune
  → Stage 5  逐场剧本生成 (scenes)      → 每场 judge → stop / prune
  → Final Judge → 选最优分支 → 输出剧本 + trace
```

默认同时维护 3 个初始种子、最多 5 个分支（含 fork）、最多 3 个活跃分支。每个 `生成 → judge → 决策` 循环都是**紧耦合的**，controller 在每个阶段之后都会运行剪枝逻辑。

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

## 5. Token 消耗模型与真实对照

Controller **理论上**通过 prune / stop 跳过后续阶段来省 token；但真实收益必须用同一 prompt、同一模型、同一 judge 的 controller vs no-controller 对照来验证。当前 Qwen3-8B + Qwen3-8B LLM judge 的 `script_001` 实验没有观察到 token 节省，默认 fork 反而增加了开销。

### 5.1 调用次数模型（默认配置）

`min_branches=3, max_branches=5, max_active_branches=3, max_scenes=4`：

- 初始种子数 = `min(max(3,1), 5)` = **3**
- Fork 上限 = `max_branches(5) − 初始(3)` = 最多 **2 次 fork**

**无 Controller 干预时**（`--disable-controller`，fork 也关闭）：

| 阶段 | Gen | Judge | 说明 |
|------|-----|-------|------|
| Stage 0 规范化 | 1 | 0 | |
| Stage 1 Seed ×3 | 3 | 3 | |
| Stage 2 Storyline ×3 | 3 | 3 | |
| Stage 3 Characters ×3 | 3 | 3 | |
| Stage 4 Outline ×3 | 3 | 3 | |
| Stage 5 Scenes ×(3×4) | 12 | 12 | |
| Final Judge ×3 | 0 | 3 | judge only |
| **合计** | **25** | **27** | **52 次** |

每次 gen 输出上限 768 token（retry 时翻倍），每次 judge 输出上限 512 token。

**每次干预的节省量**（相对 52 次上限）：

| 干预时机 | 被跳过的阶段 | 节省调用 | 占上限 |
|----------|-------------|---------|--------|
| 1 个 Seed 后 prune | Storyline + Char + Outline + 4 Scenes + Final | **15** | 29% |
| 1 个 Storyline 后 prune | Char + Outline + 4 Scenes + Final | **13** | 25% |
| 1 个 Characters 后 prune | Outline + 4 Scenes + Final | **11** | 21% |
| 1 个 Outline 后 prune | 4 Scenes + Final | **9** | 17% |
| 1 个 Scene 2 后早停 | 剩余 2 场的 gen+judge | **4** | 8% |
| Similarity dedup 1 个分支 | 同对应阶段 | 9–15 | — |

### 5.2 真实实验：Qwen3-8B + LLM Judge

实验配置：

```bash
conda run -n fhyvllm python run_pipeline.py \
  --backend hf \
  --model-path /data/fhy/models/Qwen3-8B \
  --device-map none --dtype fp16 \
  --judge-backend llm \
  --prompt-id script_001 \
  --min-branches 3 --max-branches 5 --max-active-branches 3 \
  --min-scenes 3 --max-scenes 4 \
  --max-new-tokens 1024 --scene-max-new-tokens 1536 \
  --temperature 0.55
```

同一 prompt 跑三组：默认 controller、controller 但禁用 fork、no-controller。

| 模式 | 输出目录 | Calls | Judge Calls | Tokens | Branches | Final Overall | Surprise | 选中分支 |
|------|----------|------:|------------:|-------:|---------:|--------------:|---------:|----------|
| Controller + fork | `outputs/qwen8b_llmjudge_ctl_script001` | 62 | 31 | 77,822 | 5 | 4.5 | 4.5 | b3 |
| Controller, no fork | `outputs/qwen8b_llmjudge_ctl_nofork_script001` | 52 | 26 | 66,920 | 3 | 4.5 | 4.5 | b1 |
| No-controller | `outputs/qwen8b_llmjudge_noctl_script001` | 51 | 26 | 65,395 | 3 | 4.5 | 4.5 | b3 |

相对 no-controller：

| 模式 | Calls 变化 | Token 变化 | 结论 |
|------|-----------:|-----------:|------|
| Controller + fork | +11 / +21.6% | +12,427 / +19.0% | 默认 fork 抵消并超过剪枝收益 |
| Controller, no fork | +1 / +2.0% | +1,525 / +2.3% | 几乎持平，但没有省 token |

本次真实实验不能支持“controller 默认节省 token”的结论。

### 5.3 为什么这次没有省 Token

Fork 发生在 **Storyline 阶段**（seed → storyline 展开之后，controller_prune 之前）：

```python
# pipeline.py: 所有活跃分支完成 storyline gen+judge 后
for parent in list(active):          # 遍历已有分支
    if branch_count >= max_branches: break
    if rank(parent) < 3.85: continue # 排名不够高，不 fork
    # 为高分父分支生成一个"改变冲突/反转机制"的兄弟分支
    fork_seed = call(fork_seed_prompt)    # 1 gen  — 新 seed
    judge(fork_seed)                      # 1 judge
    fork_storyline = call(storyline_prompt) # 1 gen  — 新 storyline
    judge(fork_storyline)                 # 1 judge
    # Fork 分支此时已有 seed + storyline，后续继续走 characters → outline → scenes
```

**每次 fork 立即消耗 4 次调用**（2 gen + 2 judge），然后 fork 分支被加入分支池，后续 characters / outline / scenes 阶段会继续产生调用。如果 fork 分支在 storyline 阶段就被 similarity prune 剪掉（如 trace 中的 b4），这 4 次调用就是纯浪费——fork_score_threshold 需要仔细调参来避免这种情况。

本次 `script_001` 的实际事件：

- 默认 controller 触发 2 次 fork，分支数从 3 增到 5。
- b2 在 storyline 后被 `active_prune_margin` 剪掉，b4 fork 后在 storyline 被 `controller_width_limit` 剪掉。
- b5 是从 b2 fork 出来的新分支，存活并完整跑到 scene_4。
- 最终仍有 b1、b3、b5 三个分支完整跑完 4 场，scene 生成没有减少。
- `stopped_scene_count=0`，没有触发 scene early-stop。
- no-controller 中 b2 在 scene_3 后 `no_more_scene_goals` 完成，只生成 3 场；默认 controller 反而生成了更多 scene。

禁用 fork 后也没有省 token：唯一 prune 发生在 b2 的 scene_4 之后，此时昂贵的 scene 生成已经完成，只少了最终候选评审，收益很小。

### 5.4 结论边界

当前结论只覆盖 `script_001` 单条 prompt，但它已经足够推翻 README 里原先“mock 节省可外推到真实 LLM”的表述。

- Controller 要省 token，必须在 seed/storyline/characters/outline 等早期剪掉分支，或在 scene 阶段早停；晚到 scene_4 的 prune 基本没有成本收益。
- Fork 是额外探索机制，不是节省机制。除非 fork 分支最终显著提高质量，否则默认开启 fork 可能增加 token。
- Mock 后端只能验证 pipeline 机制和理论上限，不能作为真实 token 节省证据。
- 真实节省率需要在目标模型和目标 prompt 集合上实测，并且要同时报告质量分数；只报告 calls/tokens 会掩盖 fork 是否换来了更好的结果。

### 5.5 Mock 的定位

[MockLLM](src/scriptts/llm.py#L34) 是确定性假后端：每个 stage 返回固定伪 JSON，Judge 固定返回 4.0。它适合做 smoke test、trace 检查、controller 机制回归测试；不适合论证真实 LLM 的 token 节省。

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
| `--max-branches` | 5 | 最大分支总数（含 fork） |
| `--min-branches` | 3 | 初始种子数 |
| `--max-active-branches` | 3 | 并行活跃分支上限 |
| `--max-scenes` | 4 | 最大场数 |
| `--min-scenes` | 3 | 最少场数 |
| `--max-new-tokens` | 768 | 普通阶段生成长度 |
| `--scene-max-new-tokens` | 768 | 单场生成长度 |
| `--temperature` | 0.7 | 采样温度 |
| `--fork-score-threshold` | 3.85 | 触发 fork 的排名阈值 |
| `--active-prune-margin` | 0.75 | 弱分支剪枝边距 |
| `--similarity-prune-threshold` | 0.72 | 相似分支去重阈值 |
| `--disable-controller` | false | 关闭所有 prune/stop/fork，所有分支跑满 |
| `--disable-fork` | false | 仅关闭 fork |
| `--oneshot` | false | 单次调用生成，跳过多阶段 pipeline |

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

## 8. 实验

### 8.1 真实 Token 对照（Qwen3-8B + LLM Judge）

`script_001`，同一 prompt、同一模型、同一 LLM judge，参数见 [Section 5.2](#52-真实实验qwen3-8b--llm-judge)。

| 模式 | API Calls | Tokens | Overall | Surprise | 选中分支 |
|------|----------:|-------:|--------:|---------:|----------|
| Controller + fork | 62 | 77,822 | 4.5 | 4.5 | b3 |
| Controller, no fork | 52 | 66,920 | 4.5 | 4.5 | b1 |
| No-controller | 51 | 65,395 | 4.5 | 4.5 | b3 |

观察：
- 默认 controller 没有省 token；相对 no-controller 多 12,427 tokens（+19.0%）。
- 禁用 fork 后仍没有省 token；相对 no-controller 多 1,525 tokens（+2.3%）。
- 默认 controller 与 no-controller 都选中 b3，最终评分持平；no-fork 选中 b1，但评分也持平。
- 默认 controller 的额外成本来自 2 次 fork，其中 b5 存活并完整跑到 scene_4；剪枝没有减少最终 scene 生成数量。

### 8.2 历史真实 LLM 实验（旧默认值 max_branches=2）

这些记录来自旧默认值和不同实验设置，主要用于质量与兼容性观察；它们不是当前默认配置下的 controller/no-controller token 对照。

| 配置 | API Calls | Tokens | Overall | Surprise |
|------|-----------|--------|---------|----------|
| Qwen3-8B + LLM judge v1 | 34 | 35,109 | 3.8 | 3.825 |
| Qwen3-8B + LLM judge v2 | 30 | 32,565 | 3.2 | 4.045 |
| DeepSeek v4 Pro v2 | 20 | 24,803 | 2.0 | — |

观察：
- LLM judge 比 rule judge 更能避免"AI 知道答案 = 作弊"类前提错误
- DeepSeek 在 seed/storyline/characters 阶段质量显著高于本地 8B，但 JSON 截断问题需要 retry 机制
- 4 场完整写完但结尾停留在"等待抉择"仍是主要质量问题

### 8.3 小结

- **Token 不应再用 mock 结果论证。** Mock 只说明机制上能剪枝，不能代表真实 LLM 的成本。
- **当前真实 8B 单条实验没有节省。** 默认 fork 增加探索成本，且剪枝没有早到足以跳过 scene 生成。
- **真正可能省 token 的路径**仍然是 early prune、similarity dedup、scene early-stop；但必须在目标模型和 prompt 集上实测。

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

