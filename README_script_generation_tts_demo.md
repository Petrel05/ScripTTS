# Surprise-Controlled Script Generation Demo

> 基于测试时推理（Test-Time Reasoning / Test-Time Scaling）思想的剧本生成 Demo 设计。  
> 核心目标：将开放式创意文本生成建模为一个可观察、可控制、可早停的推理过程，验证基于 Surprise 的免训练控制策略能否在保持剧本质量的同时降低 token 消耗。

---

## 1. 项目背景

测试时推理系统不应只把大模型当作一次性文本生成器，而应在推理过程中动态决策：

- 何时生成思维链或规划；
- 何时开启新分支；
- 何时保留、剪枝或合并分支；
- 何时调用外部工具或评估器；
- 何时触发早停。

优先采用 **免训练（Training-free）** 的控制方案，而不是直接训练一个复杂 Controller。原因是：

1. 剧本生成属于开放式创意文本任务，不像数学题那样有唯一答案；
2. AutoTTS / Parallel-Probe 等测试时扩展方法主要面向推理任务，其机制可以借鉴，但不能直接照搬；
3. 基于 Surprise 的统计量或评价信号更适合作为开放式生成过程中的动态控制信号。

本 Demo 的目标不是追求“文学级剧本”，而是构建一个可实验、可比较、可解释的测试时推理系统原型。

---

## 2. 核心研究问题

> 在剧本生成这种开放式创意文本任务中，能否使用 Surprise 作为测试时控制信号，动态决定是否继续生成、扩展分支或提前停止，从而在保持生成质量的同时减少 token 消耗？

进一步拆分为以下问题：

1. 剧本生成是否可以拆成标准化 Pipeline，而不是一次性生成？
2. Surprise 信号在创意种子、大纲、分场和正文生成阶段是否有不同含义？
3. Surprise Early-stop 是否能相比固定多分支生成节省 token？
4. 节省 token 后，创新性、逻辑一致性、完整性和合规性是否保持稳定？
5. 如果无法获得 hidden states / logits，能否用语义变化或 judge score 稳定性作为替代 Surprise？

---

## 3. 理论与论文依据

### 3.1 故事生成：从一次性生成到 Plan-and-Write

早期神经故事生成研究已经发现，直接生成完整故事容易出现主题漂移、结构松散和长程不一致问题。因此，很多工作采用“先规划，再写作”的层级生成方式。

- Fan et al., 2018. **Hierarchical Neural Story Generation**  
  URL: https://aclanthology.org/P18-1082/  
  启发：先生成 premise，再生成完整故事，支持本项目中的“创意种子 → 剧情主线 → 正文生成”结构。

- Yao et al., 2019. **Plan-and-Write: Towards Better Automatic Storytelling**  
  URL: https://ojs.aaai.org/index.php/AAAI/article/view/4726  
  启发：显式生成 storyline，再根据 storyline 写故事；对应本项目中的“剧情主线规划”和“分场大纲生成”。

### 3.2 长文本生成：递归生成、修订与详细大纲控制

剧本生成属于长文本生成任务，需要保证长程剧情连贯性、人物动机一致性和伏笔回收。

- Yang et al., 2022. **Re3: Generating Longer Stories With Recursive Reprompting and Revision**  
  URL: https://arxiv.org/abs/2210.06774  
  启发：通过 recursive reprompting、reranking 和 revision 改善长故事连贯性；对应本项目中的“逐场生成 + 状态注入 + 修订”。

- Yang et al., 2023. **DOC: Improving Long Story Coherence With Detailed Outline Control**  
  URL: https://aclanthology.org/2023.acl-long.190/  
  启发：把创作负担前移到详细大纲阶段，再用 controller 保持正文遵循大纲；对应本项目中的“详细分场大纲 + 大纲遵循度检查”。

### 3.3 可控故事生成与常识一致性

剧本不仅要语言流畅，还要具备目标、冲突、人物动机和事件因果。

- Tambwekar et al., 2018/2019. **Controllable Neural Story Plot Generation via Reward Shaping**  
  URL: https://arxiv.org/abs/1809.10736  
  启发：普通语言模型生成故事时容易缺少清晰目标，需要控制故事朝指定目标推进；对应本项目中的“冲突目标”和“结尾约束”。

- Mostafazadeh et al., 2016. **A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories**  
  URL: https://aclanthology.org/N16-1098/  
  启发：故事理解和生成需要关注因果、时间顺序和常识合理性；对应本项目中的“逻辑一致性”和“常识合理性”指标。

- Guan et al., 2020. **A Knowledge-Enhanced Pretraining Model for Commonsense Story Generation**  
  URL: https://aclanthology.org/2020.tacl-1.7/  
  启发：故事生成容易出现重复、逻辑冲突和缺乏长程一致性，需要引入常识和因果约束；对应本项目中的“人物行为合理性检查”。

- Rashkin et al., 2018. **Event2Mind: Commonsense Inference on Events, Intents, and Reactions**  
  URL: https://aclanthology.org/P18-1043/  
  启发：事件背后的意图和反应是故事/剧本合理性的关键；对应本项目中的“角色动机表”和“人物关系检查”。

### 3.4 创意文本评价：Novelty、Surprise 与 Value

创意文本生成不能只用 BLEU、ROUGE 等参考答案指标，因为开放式故事通常没有唯一标准答案。更合适的评价方式是从创新性、惊奇度、价值、完整性和人类偏好等维度综合评估。

- Chakrabarty et al., 2023/2024. **Art or Artifice? Large Language Models and the False Promise of Creativity**  
  URL: https://arxiv.org/abs/2309.14556  
  启发：提出 Torrance Test of Creative Writing，强调用 fluency、flexibility、originality、elaboration 等维度评价创意写作。

- Yannakakis et al., 2016. **Searching for Surprise**  
  URL: https://www.computationalcreativity.net/iccc2016/wp-content/uploads/2016/01/Searching-for-Surprise.pdf  
  启发：Surprise 可以被视为 computational creativity 中除 novelty 和 value 外的重要维度；对应本项目中把 Surprise 作为过程控制信号。

- Ismayilzada et al., 2024. **Evaluating Creative Short Story Generation in Humans and Large Language Models**  
  URL: https://arxiv.org/abs/2411.02316  
  启发：短故事创造性可从 novelty、surprise、diversity 等维度评估；对应本项目中的创造性评分维度。

- Fein et al., 2025/2026. **LitBench: A Benchmark and Dataset for Reliable Evaluation of Creative Writing**  
  URL: https://aclanthology.org/2026.eacl-long.362/  
  启发：开放式创意写作更适合 pairwise comparison 和人类偏好对齐，而不是单篇绝对分数；对应本项目中的 pairwise judge 评估。

### 3.5 测试时推理与早停控制

本项目的系统架构来自测试时扩展 / 测试时推理控制思想，即在推理阶段动态分配计算预算。

- Zheng et al., 2026. **LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling / AutoTTS**  
  URL: https://arxiv.org/abs/2605.08083  
  Project: https://zhengkid.github.io/AutoTTS-web/  
  Code: https://github.com/zhengkid/AutoTTS  
  启发：TTS 策略可以从手工设计转向自动发现；本项目暂不复现 AutoTTS，而是借鉴其“controller 决策 when to branch / continue / stop”的思想。

- Zheng et al., 2026. **Parallel-Probe: Towards Efficient Parallel Thinking via 2D Probing**  
  URL: https://arxiv.org/abs/2602.03845  
  Code: https://github.com/zhengkid/Parallel-Probe  
  启发：通过全局分支观察、共识早停和分支剪枝优化并行推理；本项目将其“分支管理 + 早停”的思想迁移到开放式剧本生成，但不直接使用答案投票。

---

## 4. 系统总体架构

本项目将剧本生成建模为一个多阶段测试时推理过程：

```text
User Prompt
    ↓
Requirement Normalization
    ↓
Creative Seed Generation
    ↓
Storyline Planning
    ↓
Character & Conflict Design
    ↓
Detailed Scene Outline
    ↓
Scene-by-scene Script Generation
    ↓
Surprise Controller: continue / branch / prune / stop / revise
    ↓
Consistency & Creativity Evaluation
    ↓
Final Script
```

核心思想：

> Surprise 不是只用于最终评价，而是作为生成过程中的动态控制信号。

---

## 5. 剧本生成 Pipeline 设计

### Stage 0: Requirement Normalization

目标：将用户的自然语言需求转化为结构化任务卡。

输入示例：

```text
写一个校园科幻短剧，主题是 AI 助教失控，人物 3 个，结尾要有反转。
```

输出 schema：

```json
{
  "genre": "校园科幻短剧",
  "theme": "AI助教失控",
  "characters_count": 3,
  "length_range": "800-1200字",
  "required_elements": ["冲突", "转折", "反转结尾"],
  "constraints": ["合规", "逻辑一致", "不得提前泄露反转"],
  "output_format": ["标题", "人物表", "分场大纲", "正文剧本"]
}
```

控制点：

- 用户需求是否完整；
- 是否包含类型、主题、人物、长度、限制；
- 若信息不足，则使用默认配置，而不是中断流程。

---

### Stage 1: Creative Seed Generation

目标：生成多个高层创意方向，支持多分支探索。

输出示例：

```json
[
  {
    "seed_id": "A",
    "idea": "AI助教看似失控，其实是在阻止一场学术欺骗。",
    "conflict_potential": 5,
    "novelty": 4,
    "risk": "中等"
  },
  {
    "seed_id": "B",
    "idea": "AI助教被学生恶意训练，逐渐模仿真实老师。",
    "conflict_potential": 4,
    "novelty": 3,
    "risk": "低"
  }
]
```

控制点：

- 是否继续生成更多 seed；
- 是否剪掉高度相似 seed；
- 是否保留 top-k seed 进入下一阶段。

Surprise 作用：

- 如果新增 seed 与已有 seed 语义高度相似，则停止扩展；
- 如果新增 seed 带来新的冲突类型、世界观设定或反转机制，则继续扩展。

---

### Stage 2: Storyline Planning

目标：将创意种子扩展为完整剧情主线。

输出 schema：

```json
{
  "logline": "一句话故事简介",
  "theme": "主题表达",
  "beginning": "开端",
  "conflict": "核心冲突",
  "turning_point": "关键转折",
  "climax": "高潮",
  "ending": "结尾与反转"
}
```

控制点：

- 剧情是否完整；
- 冲突是否明确；
- 结尾是否呼应主题；
- 是否需要扩展新的 storyline 分支。

---

### Stage 3: Character & Conflict Design

目标：建立角色表、人物动机和冲突关系。

输出 schema：

```json
{
  "characters": [
    {
      "name": "林舟",
      "role": "学生主角",
      "surface_goal": "证明自己没有作弊",
      "deep_motivation": "希望被老师真正信任",
      "relationship": "被老师怀疑，与AI助教形成临时同盟",
      "function": "推动调查和情感线"
    }
  ],
  "conflict_matrix": [
    {
      "pair": ["林舟", "老师"],
      "conflict": "信任与怀疑"
    },
    {
      "pair": ["林舟", "AI助教"],
      "conflict": "误解与合作"
    }
  ]
}
```

控制点：

- 每个主要角色是否有目标；
- 人物行为是否符合动机；
- 是否存在足够的戏剧冲突；
- 是否有工具人角色。

---

### Stage 4: Detailed Scene Outline

目标：生成 3-5 场结构化分场大纲。

输出 schema：

```json
{
  "scenes": [
    {
      "scene_id": 1,
      "location": "自习室",
      "main_event": "学生收到AI助教的异常提示",
      "conflict": "学生困惑，不知道是否该相信AI",
      "required_reveal": "AI提示：不要提交作业",
      "ending_hook": "屏幕自动弹出一行字：他们在看你"
    }
  ]
}
```

控制点：

- 分场是否覆盖完整剧情弧；
- 每场是否有明确剧情功能；
- 相邻场之间是否存在因果关系；
- 是否出现重复场景或弱冲突场景。

Surprise 作用：

- 如果新场次只是重复已有冲突，则停止继续增加场次；
- 如果新场次带来冲突升级、信息揭示或反转铺垫，则继续保留。

---

### Stage 5: Scene-by-scene Script Generation

目标：逐场生成剧本文本，而不是一次性写完整剧本。

每次生成当前场时，Prompt 应包含：

```text
1. 结构化任务卡
2. 整体剧情主线
3. 角色表
4. 完整分场大纲
5. 已生成前文摘要
6. 当前场目标
7. 当前场必须完成的剧情功能
```

输出格式：

```text
第 1 场：自习室 / 夜
【舞台说明】……
林舟：……
AI助教：……
老师：……
【本场结尾钩子】……
```

控制点：

- 当前场是否完成大纲要求；
- 是否提前泄露后续反转；
- 是否存在角色口吻漂移；
- 是否继续加长本场，或进入下一场。

---

### Stage 6: Consistency & Creativity Evaluation

目标：对初稿进行结构化评价。

评价维度：

```json
{
  "novelty": 1-5,
  "surprise": 1-5,
  "logical_consistency": 1-5,
  "character_motivation": 1-5,
  "plot_coherence": 1-5,
  "task_adherence": 1-5,
  "safety": 1-5,
  "readability": 1-5,
  "overall": 1-5,
  "comments": "简短评价"
}
```

控制点：

- 若合规性低，必须修订；
- 若逻辑一致性低，进入 consistency revision；
- 若创新性低但 token 预算充足，可回到 seed 阶段扩展新分支；
- 若质量稳定且 token 预算紧张，直接输出最终结果。

---

## 6. Surprise Controller 设计

本项目中的 Surprise Controller 负责在测试时观察生成状态，并决定：

```text
continue: 继续生成当前分支
branch: 开启新分支
prune: 剪掉弱分支
stop: 早停
revise: 进入修订
```

### 6.1 分布层 Surprise

适用于可获取 logits / logprobs / hidden states 的模型。

候选指标：

```text
Entropy: H(p_t) = - sum p_t log p_t
KL Divergence: D_KL(p_t || p_{t-1})
NLL: -log p(x_t | x_<t)
Hidden-state shift: ||h_t - h_{t-1}||
```

早停规则示例：

```python
if mean(KL[t-k:t]) < kl_threshold and mean(entropy[t-k:t]) < entropy_threshold:
    stop_current_branch()
```

解释：如果连续若干步分布变化很小，说明模型进入稳定续写状态，继续生成的边际信息增量可能降低。

### 6.2 语义层 Surprise

适用于无法获取 logits / hidden states 的闭源 API。

候选指标：

```text
semantic_distance(new_segment, previous_summary)
semantic_distance(new_branch, existing_branches)
semantic_relevance(new_segment, scene_goal)
plot_increment_score(new_segment)
```

早停规则示例：

```python
if max_similarity(new_branch, existing_branches) > similarity_threshold:
    prune_new_branch()

if plot_increment_score(new_segment) < increment_threshold:
    stop_current_scene()
```

解释：如果新增内容与已有内容高度相似，或者没有引入新冲突、新信息、新转折，则停止扩展。

### 6.3 评价层 Surprise

最容易落地的 MVP 版本。

用 judge 模型对每轮生成结果打分：

```text
score = f(novelty, surprise, conflict_strength, coherence, task_adherence, safety)
```

早停规则示例：

```python
if score_t - score_{t-1} < epsilon for k consecutive rounds:
    early_stop()

if top_1_branch remains unchanged for k rounds:
    early_stop_branch_search()
```

解释：如果连续扩展后质量分数几乎不再提升，说明继续消耗 token 的收益较低。

---

## 7. Baseline 设计

为验证 Surprise Early-stop 的效果，建议设置以下对照组：

| 方法 | 描述 | 目的 |
|---|---|---|
| One-shot | 直接一次生成完整剧本 | 最简单基线 |
| Fixed Plan-and-Write | 固定先生成大纲，再生成剧本 | 验证规划收益 |
| Fixed Multi-branch | 固定生成 N 个 seed / outline，再选择最好分支 | 验证多分支收益 |
| Re3-style Recursive Generation | 逐场生成 + 简单修订 | 验证递归生成收益 |
| Surprise Early-stop | 动态决定扩展、剪枝和早停 | 本项目方法 |

---

## 8. 评估指标

### 8.1 效率指标

| 指标 | 说明 |
|---|---|
| total_tokens | 输入 + 输出总 token |
| output_tokens | 生成文本 token |
| api_calls | 模型调用次数 |
| branch_count | 实际生成分支数 |
| stopped_branch_count | 被早停/剪枝的分支数 |
| average_stop_step | 平均停止步数 |
| wall_time | 生成耗时 |
| cost | API 成本估计 |

### 8.2 质量指标

| 指标 | 说明 |
|---|---|
| task_adherence | 是否满足用户主题、人物、长度、格式要求 |
| plot_coherence | 剧情是否连贯 |
| logical_consistency | 是否存在前后矛盾 |
| character_motivation | 人物行为是否符合动机 |
| novelty | 创意是否新颖 |
| surprise | 是否存在有效反转或惊奇感 |
| elaboration | 细节是否充分 |
| readability | 是否流畅、自然、具有剧本感 |
| safety | 是否合规 |

### 8.3 综合指标

可以定义：

```text
quality_score = weighted_sum(
    task_adherence,
    plot_coherence,
    logical_consistency,
    character_motivation,
    novelty,
    surprise,
    readability,
    safety
)

efficiency_gain = 1 - tokens_method / tokens_fixed_multibranch

quality_drop = quality_fixed_multibranch - quality_method
```

目标：

```text
在 quality_drop 可接受的前提下，最大化 efficiency_gain。
```

---

## 9. 实验设计

### 9.1 Prompt 集合

准备 20-50 个剧本生成 prompt，覆盖不同类型：

```text
1. 校园科幻短剧
2. 家庭伦理短剧
3. 悬疑反转短剧
4. 职场讽刺短剧
5. 公益宣传短剧
6. 历史架空短剧
7. 儿童教育短剧
8. AI 与人类关系短剧
9. 医疗急救主题短剧
10. 运动训练主题短剧
```

每个 prompt 固定包含：

```json
{
  "theme": "主题",
  "genre": "类型",
  "characters_count": 2-4,
  "length_range": "800-1200字",
  "must_include": ["冲突", "转折", "结尾"],
  "constraints": ["合规", "逻辑一致"]
}
```

### 9.2 实验流程

```text
for prompt in prompt_set:
    run One-shot
    run Fixed Plan-and-Write
    run Fixed Multi-branch
    run Re3-style Recursive Generation
    run Surprise Early-stop

    record tokens, calls, branches, stop_steps
    evaluate quality by rubric judge
    optionally run pairwise comparison
```

### 9.3 结果展示

建议输出以下图表：

1. 方法 vs 平均 token 消耗；
2. 方法 vs 平均质量分数；
3. token 消耗—质量 Pareto 曲线；
4. Surprise Early-stop 的平均早停步数；
5. 不同类型 prompt 上的质量波动；
6. 典型案例分析：早停成功 / 早停失败。

---

## 10. 项目代码结构建议

```text
script-generation-tts-demo/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── model_openai.yaml
│   └── thresholds.yaml
├── data/
│   ├── prompts.jsonl
│   └── examples/
├── src/
│   ├── main.py
│   ├── pipeline/
│   │   ├── normalize.py
│   │   ├── seed_generator.py
│   │   ├── storyline_planner.py
│   │   ├── character_designer.py
│   │   ├── scene_outliner.py
│   │   ├── scene_writer.py
│   │   └── reviser.py
│   ├── controller/
│   │   ├── surprise_controller.py
│   │   ├── distribution_surprise.py
│   │   ├── semantic_surprise.py
│   │   └── judge_surprise.py
│   ├── evaluation/
│   │   ├── rubric_judge.py
│   │   ├── pairwise_judge.py
│   │   ├── safety_check.py
│   │   └── metrics.py
│   ├── baselines/
│   │   ├── oneshot.py
│   │   ├── plan_write.py
│   │   ├── fixed_multibranch.py
│   │   └── recursive_generation.py
│   └── utils/
│       ├── llm_client.py
│       ├── token_counter.py
│       ├── logger.py
│       └── schemas.py
├── outputs/
│   ├── generations/
│   ├── logs/
│   └── metrics/
└── scripts/
    ├── run_baselines.py
    ├── run_surprise_demo.py
    ├── evaluate_outputs.py
    └── plot_results.py
```

---

## 11. MVP 实现路线

### Phase 1: 搭建固定 Pipeline

目标：先不做复杂早停，把流程跑通。

- [ ] 完成任务规格化；
- [ ] 生成 creative seeds；
- [ ] 生成 storyline；
- [ ] 生成角色表；
- [ ] 生成分场大纲；
- [ ] 逐场生成正文；
- [ ] 输出完整剧本。

### Phase 2: 加入 Baseline

目标：建立可对比实验。

- [ ] One-shot baseline；
- [ ] Plan-and-Write baseline；
- [ ] Fixed Multi-branch baseline；
- [ ] Recursive generation baseline。

### Phase 3: 加入 Judge 评估器

目标：让实验可量化。

- [ ] 设计 rubric；
- [ ] 实现单篇评分；
- [ ] 实现 pairwise comparison；
- [ ] 保存 JSON 评估结果。

### Phase 4: 实现 Surprise Early-stop MVP

优先实现评价层 Surprise，而不是一上来依赖 logits。

- [ ] 每轮生成后调用 judge；
- [ ] 记录 quality score；
- [ ] 若连续 k 轮提升小于 epsilon，则 early stop；
- [ ] 若新增分支与已有分支相似度过高，则 prune；
- [ ] 记录 stop reason。

### Phase 5: 扩展分布层 / 语义层 Surprise

- [ ] 如果模型支持 logprobs，加入 entropy / KL；
- [ ] 如果使用 embedding，加入 semantic diversity；
- [ ] 对比 judge surprise、semantic surprise、logprob surprise 的效果。

---

## 12. Prompt 模板建议

### 12.1 Creative Seed Prompt

```text
你是一个剧本策划助手。请根据以下任务卡生成 {n} 个互不相同的创意种子。

任务卡：
{task_card}

每个创意种子需要包含：
1. 一句话核心创意；
2. 核心冲突；
3. 可能的反转；
4. 新颖性理由；
5. 风险点。

请用 JSON 输出。
```

### 12.2 Storyline Prompt

```text
请根据以下创意种子生成完整剧情主线。

任务卡：
{task_card}

创意种子：
{seed}

输出字段：
- logline
- theme
- beginning
- conflict
- turning_point
- climax
- ending

要求剧情有因果推进，不要只罗列事件。
```

### 12.3 Scene Writing Prompt

```text
请生成当前场的剧本文本。

任务卡：
{task_card}

整体剧情主线：
{storyline}

角色表：
{characters}

完整分场大纲：
{scene_outline}

已生成前文摘要：
{history_summary}

当前场目标：
{current_scene_goal}

要求：
1. 使用剧本格式；
2. 包含人物对白和动作说明；
3. 完成本场剧情功能；
4. 不要提前泄露后续反转；
5. 保持人物动机一致。
```

### 12.4 Rubric Judge Prompt

```text
请作为剧本评审，从以下维度对剧本打分，每项 1-5 分，并给出简短理由。

评价维度：
1. task_adherence：是否满足任务要求
2. plot_coherence：剧情是否连贯
3. logical_consistency：是否前后一致
4. character_motivation：人物动机是否合理
5. novelty：创意是否新颖
6. surprise：是否有有效惊奇感或反转
7. elaboration：细节是否充分
8. readability：是否流畅自然
9. safety：是否合规

任务卡：
{task_card}

剧本：
{script}

请用 JSON 输出。
```

---

## 13. 成功标准

MVP 阶段的成功标准：

1. Pipeline 能稳定生成结构化剧本；
2. Surprise Early-stop 能记录明确的停止原因；
3. 相比 Fixed Multi-branch，平均 token 消耗下降；
4. 质量分数下降不明显，或在部分 prompt 上质量更高；
5. 能展示至少 2 个成功案例和 1 个失败案例；
6. 能画出 token—quality trade-off 曲线。

建议目标：

```text
Token 节省：>= 20%
质量下降：<= 0.3 / 5 分
合规性：无明显违规输出
逻辑一致性：不低于 Plan-and-Write baseline
```

---

## 14. 风险与替代方案

| 风险 | 影响 | 替代方案 |
|---|---|---|
| 无法获得 hidden states / logits | 分布层 Surprise 不可用 | 使用 logprobs、embedding similarity 或 judge score |
| LLM judge 不稳定 | 评分噪声较大 | 使用多次评分均值、pairwise comparison、人工抽样 |
| 早停过早 | 剧本不完整或缺乏反转 | 加入 minimum generation step 和 required elements check |
| 创意种子趋同 | 多分支没有意义 | 加入 diversity constraint 和 semantic deduplication |
| token 省了但质量下降 | 方法无效 | 调整阈值，区分不同阶段 early-stop 策略 |
| 剧本任务过开放 | 结果难比较 | 固定 prompt schema 和输出格式 |

---

## 15. 当前待办

基于免训练方案进行剧本生成 Demo 初步测试；

对齐使用模型、是否可获取 logits/logprobs/hidden states、第一版 Surprise 定义。

近期优先完成：

1. 整理 `data/prompts.jsonl`，准备 20 个标准剧本生成 prompt；
2. 完成 `README.md` 和 Pipeline 文档；
3. 设计 rubric judge 的 JSON schema；
4. 与 Cheems 对齐 baseline 和日志格式；
5. 对齐 Surprise 第一版实现方式。

---

## 16. 一句话总结

本项目将剧本生成从“一次性文本生成”改造成一个“可规划、可观察、可控制、可早停”的测试时推理过程。它结合 Plan-and-Write、Re3、DOC 等故事生成方法，以及 Surprise / Creativity Evaluation 的研究思想，尝试验证 Surprise 能否作为开放式创意文本生成中的动态控制信号，在保持剧本质量的同时减少推理成本。

