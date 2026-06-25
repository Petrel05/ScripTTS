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

**关键特性**：每个 `生成 → judge → 决策` 循环都是**紧耦合的**，controller 在每个阶段之后都会运行剪枝逻辑，而不是等所有内容生成完再回头挑选。

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

## 5. Token 节省分析

### 省 Token 机制

| 机制 | 原理 | 节省量级 |
|------|------|---------|
| **Scene Stop** | 质量够了就停，不写满所有场 | 每场 ~768 token 输出 + prompt |
| **Branch Prune** | 低分/相似分支直接杀死 | 省掉后续全部 stage 的 token |
| **Similarity Dedup** | 相似分支只留一个 | 避免重复生成 |
| **Rescue 保底** | 正常情况不触发 | 避免空跑浪费 |

### 定量估算（默认配置）

- **最坏情况（无剪枝）**：~16 次 LLM 调用
- **典型情况（有剪枝）**：节省 25–50% 调用
- **Judge 开销**：每次 `max_new_tokens=512`（6 维 JSON 很短），可控

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
