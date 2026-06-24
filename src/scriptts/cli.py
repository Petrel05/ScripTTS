from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .data import read_jsonl, write_jsonl
from .llm import HFLocalLLM, MockLLM
from .pipeline import PipelineConfig, ScriptPipeline, save_result_markdown


DEFAULT_MODEL_ROOT = "/data/fhy/models"
DEFAULT_MODEL_NAME = "Qwen3-0.6B"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    tasks = read_jsonl(input_path)
    if args.prompt_id:
        tasks = [task for task in tasks if task.id == args.prompt_id]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("No tasks selected.")

    llm = build_llm(args)
    config = PipelineConfig(
        max_branches=args.max_branches,
        max_scenes=args.max_scenes,
        min_scenes=args.min_scenes,
        max_new_tokens=args.max_new_tokens,
        scene_max_new_tokens=args.scene_max_new_tokens,
        temperature=args.temperature,
        judge_backend=args.judge_backend,
    )
    pipeline = ScriptPipeline(llm=llm, config=config)

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_name
    scripts_dir = output_dir / "scripts"
    records = []

    print(f"[ScripTTS] selected_tasks={len(tasks)} output_dir={output_dir}")
    for task in tasks:
        print(f"[ScripTTS] running {task.id}: {task.theme}")
        result = pipeline.run_task(task)
        records.append(result.record)
        save_result_markdown(scripts_dir / f"{result.task_id}.md", result)
        metrics = result.record["metrics"]
        score = result.record["final_score"]
        print(
            "[ScripTTS] done "
            f"{task.id} overall={score['overall_quality']} surprise={score['useful_surprise']} "
            f"calls={metrics['api_calls']} tokens={metrics['total_tokens']}"
        )

    write_jsonl(output_dir / "results.jsonl", records)
    print(f"[ScripTTS] wrote {output_dir / 'results.jsonl'}")
    print(f"[ScripTTS] wrote scripts to {scripts_dir}")


def build_llm(args: argparse.Namespace):
    if args.backend == "mock":
        return MockLLM()

    model_path = args.model_path
    if not model_path:
        model_path = str(Path(DEFAULT_MODEL_ROOT) / DEFAULT_MODEL_NAME)

    if args.backend == "auto":
        try:
            return HFLocalLLM(
                model_path=model_path,
                device_map=args.device_map,
                dtype=args.dtype,
                local_files_only=not args.allow_remote_files,
            )
        except Exception as exc:
            print(f"[ScripTTS] HF backend unavailable, falling back to mock: {exc}")
            return MockLLM()

    return HFLocalLLM(
        model_path=model_path,
        device_map=args.device_map,
        dtype=args.dtype,
        local_files_only=not args.allow_remote_files,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal surprise-controlled script generation pipeline.")
    parser.add_argument("--input", default="prompts.jsonl", help="Input jsonl file.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory root.")
    parser.add_argument("--run-name", default="", help="Optional stable run directory name.")
    parser.add_argument("--prompt-id", default="", help="Run a single task id, e.g. script_001.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks.")

    parser.add_argument("--backend", choices=["auto", "hf", "mock"], default="auto")
    parser.add_argument("--model-path", default="", help="Local HF model path. Default: /data/fhy/models/Qwen3-0.6B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--allow-remote-files", action="store_true", help="Allow transformers to fetch missing files.")

    parser.add_argument("--judge-backend", choices=["rule", "llm"], default="rule")
    parser.add_argument("--max-branches", type=int, default=2)
    parser.add_argument("--max-scenes", type=int, default=4)
    parser.add_argument("--min-scenes", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--scene-max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()
