# ScripTTS MVP 快速运行

这个版本先实现一条能跑通的最小流程：

```text
prompts.jsonl
  -> 需求规范化
  -> 创意种子分支
  -> 选择分支
  -> 剧情主线
  -> 人物与冲突
  -> 分场大纲
  -> 逐场剧本
  -> rule/LLM judge
  -> results.jsonl + scripts/*.md
```

Surprise 暂时是轻量版本：

```text
UsefulSurprise = 0.35 * Novelty + 0.35 * PlotProgress + 0.30 * Relevance
```

规则早停采用 `README_surprise.md` 中的第一版思想：连续低 surprise、低质量增益且候选仍有效时，记录 stop 决策。

## 1. 本地 smoke test

不需要模型依赖，用 mock 后端检查流程：

```bash
python3 run_pipeline.py --backend mock --limit 1 --run-name smoke_mock
```

输出：

```text
outputs/smoke_mock/results.jsonl
outputs/smoke_mock/scripts/script_001.md
```

## 2. 远程服务器运行

假设项目同步到：

```text
/home/fhy/ScripTTS
```

模型目录：

```text
/data/fhy/models/DeepSeek-R1-Distill-Qwen-7B
/data/fhy/models/Qwen3-0.6B
/data/fhy/models/Qwen3-1.7B
/data/fhy/models/Qwen3-4B
/data/fhy/models/Qwen3-8B
```

创建 conda 环境并安装基础依赖：

```bash
cd /home/fhy/ScripTTS
conda create -n scriptts python=3.10 -y
conda activate scriptts
pip install torch transformers accelerate
```

先用最小模型跑单条：

```bash
python3 run_pipeline.py \
  --backend hf \
  --model-path /data/fhy/models/Qwen3-0.6B \
  --prompt-id script_001 \
  --run-name qwen06b_one
```

跑全部 20 条：

```bash
python3 run_pipeline.py \
  --backend hf \
  --model-path /data/fhy/models/Qwen3-0.6B \
  --run-name qwen06b_all
```

切换大模型只需要换 `--model-path`：

```bash
python3 run_pipeline.py \
  --backend hf \
  --model-path /data/fhy/models/Qwen3-8B \
  --prompt-id script_001 \
  --run-name qwen8b_one
```

## 3. Judge 模式

默认使用规则 judge，稳定、快，适合先验证全流程：

```bash
python3 run_pipeline.py --backend hf --model-path /data/fhy/models/Qwen3-0.6B --judge-backend rule --limit 3
```

也可以让同一个 LLM 做 judge：

```bash
python3 run_pipeline.py --backend hf --model-path /data/fhy/models/Qwen3-0.6B --judge-backend llm --limit 3
```

如果 LLM judge 输出不是合法 JSON，代码会自动退回规则 judge。

## 4. 常用参数

- `--limit 3`：只跑前三条。
- `--prompt-id script_001`：只跑指定 prompt。
- `--max-branches 2`：创意种子分支数量。
- `--max-scenes 4`：最多逐场生成多少场。
- `--max-new-tokens 768`：普通阶段生成长度。
- `--scene-max-new-tokens 768`：单场剧本生成长度。
- `--temperature 0.7`：采样温度。
- `--output-dir outputs`：输出根目录。
- `--run-name NAME`：固定本次输出目录名。

## 5. 输出字段

`results.jsonl` 每行对应一个任务，包含：

- `task`：原始输入任务。
- `task_card`：需求规范化结果。
- `branches`：创意分支、分支评分、分支决策。
- `chosen_seed`：选中的种子。
- `storyline`：剧情主线。
- `characters`：人物与冲突。
- `outline`：分场大纲。
- `scenes`：逐场剧本、每场评分、早停决策。
- `final_score`：最终规则或 LLM judge 分数。
- `metrics`：调用次数、估算 token、分支数、耗时等。

`scripts/*.md` 是方便人工阅读的最终剧本。
