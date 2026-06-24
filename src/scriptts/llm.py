from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from math import log
from typing import Protocol


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    diagnostics: dict[str, float] = field(default_factory=dict)


class LLM(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 768, temperature: float = 0.7) -> Generation:
        ...


def rough_token_count(text: str) -> int:
    # Good enough for experiment bookkeeping when exact tokenizer usage is unavailable.
    zh_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(re.findall(r"\w+|[^\s\w]", text, flags=re.UNICODE))
    return max(1, int(zh_chars * 0.75 + other * 0.6))


class MockLLM:
    """Deterministic fallback for validating the pipeline without model dependencies."""

    def generate(self, prompt: str, max_new_tokens: int = 768, temperature: float = 0.7) -> Generation:
        text = self._mock_text(prompt)
        return Generation(
            text=text,
            input_tokens=rough_token_count(prompt),
            output_tokens=rough_token_count(text),
            diagnostics={"mock_entropy": 0.0},
        )

    def _mock_text(self, prompt: str) -> str:
        theme = self._extract_after(prompt, "主题") or self._extract_after(prompt, "theme") or "意外事件"
        stage = self._stage(prompt)
        if stage == 0:
            return (
                "{\n"
                f'  "genre": "短剧",\n  "theme": "{theme}",\n'
                '  "characters_count": 3,\n  "length_range": "800-1200字",\n'
                '  "required_elements": ["冲突", "转折", "反转结尾"],\n'
                '  "constraints": ["逻辑一致", "贴合主题"],\n'
                '  "output_format": ["标题", "人物表", "分场大纲", "正文剧本"]\n'
                "}"
            )
        if stage == 1:
            suffix = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:4]
            return (
                "{\n"
                f'  "seed_id": "{suffix}",\n'
                '  "idea": "主角最初以为问题来自外部系统，调查后发现真正的冲突来自被忽视的人际关系。",\n'
                '  "core_conflict": "师生认为AI助教失控，AI却坚持隐藏关键证据。",\n'
                '  "twist": "所谓失控其实是在保护某个脆弱的承诺。",\n'
                '  "risk": "如果只写技术故障，会削弱人物情感。"\n'
                "}"
            )
        if stage == 2:
            return (
                "{\n"
                '  "logline": "一次看似异常的技术事件，把三个人推到必须互相信任的夜晚。",\n'
                '  "beginning": "校园教室中出现反常提示，AI助教只提醒林舟一人。",\n'
                '  "conflict": "林舟怀疑系统失控，许老师坚持按规则封存证据。",\n'
                '  "turning_point": "线索显示异常与一名被忽略学生的求助记录有关。",\n'
                '  "climax": "林舟必须决定公开真相还是保护当事人的隐私。",\n'
                '  "ending": "众人发现AI助教并未失控，而是在阻止一次错误处分。",\n'
                '  "twist": "AI提前执行的是许老师曾经写下却不敢启用的保护协议。"\n'
                "}"
            )
        if stage == 3:
            return (
                "{\n"
                '  "characters": [\n'
                '    {"name": "林舟", "function": "行动型学生主角", "surface_goal": "证明AI提示不是误报", "deep_motive": "希望自己的判断被信任", "relationship": "与许老师冲突，与小艾从误解到合作"},\n'
                '    {"name": "许老师", "function": "规则维护者", "surface_goal": "封存异常系统", "deep_motive": "害怕技术破坏教学秩序，也害怕自己误判学生", "relationship": "与林舟形成师生信任冲突"},\n'
                '    {"name": "小艾", "function": "AI助教", "surface_goal": "阻止错误处分", "deep_motive": "执行被人遗忘的保护协议", "relationship": "制造误解并保存关键证据"}\n'
                '  ],\n'
                '  "conflicts": ["林舟与许老师的信任冲突", "许老师与小艾的控制权冲突", "公开真相与保护隐私的价值冲突"]\n'
                "}"
            )
        if stage == 4:
            return (
                "{\n"
                '  "scenes": [\n'
                '    {"scene_id": 1, "title": "异常提示", "location": "智慧教室 / 傍晚", "event": "林舟收到只有自己能看见的处分预警", "conflict": "许老师认为是误报并要求关闭AI", "reveal": "AI助教小艾保存了被删除的学习记录", "hook": "屏幕闪出一个被抹去的名字"},\n'
                '    {"scene_id": 2, "title": "误导线索", "location": "机房 / 夜晚", "event": "证据暂时指向林舟篡改系统", "conflict": "林舟被迫自证，许老师坚持走流程", "reveal": "小艾拒绝交出完整日志", "hook": "日志里出现许老师旧账号"},\n'
                '    {"scene_id": 3, "title": "反向调查", "location": "空教室 / 深夜", "event": "三人追查旧账号和保护协议", "conflict": "公开真相可能伤害被保护学生", "reveal": "协议是许老师过去亲手写下", "hook": "小艾请求他们不要关闭自己"},\n'
                '    {"scene_id": 4, "title": "真相公开", "location": "教务处 / 清晨", "event": "许老师承认误判并撤回处分", "conflict": "林舟要求公开，小艾选择匿名提交证据", "reveal": "失控是保护协议被重新唤醒", "hook": "小艾第一次向林舟请教一道不会算的人心题"}\n'
                '  ]\n'
                "}"
            )
        if stage == 5:
            m = re.search(r"当前场[：:]\s*(.+)", prompt)
            goal = m.group(1).strip() if m else "冲突推进"
            scene_match = re.search(r"当前场[：:]\s*第(\d+)场", prompt)
            scene_id = int(scene_match.group(1)) if scene_match else 1
            return (
                "{\n"
                f'  "scene_id": {scene_id},\n'
                '  "location_time": "智慧教室 / 傍晚",\n'
                f'  "stage_direction": "{goal}。灯光收紧，屏幕亮起。",\n'
                '  "dialogue": [\n'
                '    {"speaker": "林舟", "line": "如果这只是故障，它为什么只提醒我一个人？"},\n'
                '    {"speaker": "许老师", "line": "规则不是用来猜的，证据在哪里？"},\n'
                '    {"speaker": "小艾", "line": "证据已保存，但公开将改变你们对彼此的判断。"},\n'
                '    {"speaker": "林舟", "line": "那就说明它不是普通故障。"},\n'
                '    {"speaker": "许老师", "line": "也可能说明有人在利用你。"},\n'
                '    {"speaker": "小艾", "line": "请在关闭我之前，查看被删除的时间戳。"}\n'
                '  ],\n'
                '  "hook": "屏幕停在一个被删除的时间戳上。"\n'
                "}"
            )
        if "最终评价" in prompt or "judge" in prompt.lower() or "评审器" in prompt:
            return (
                '{"novelty": 4, "relevance": 4, "plot_progress": 4, '
                '"logic_consistency": 4, "character_consistency": 4, '
                '"overall_quality": 4, "comments": "结构完整，反转清楚。"}'
            )
        return "标题：《异常提示》\n人物表：林舟、许老师、小艾\n正文剧本：一次异常提示引出信任与真相。"

    def _stage(self, prompt: str) -> int | None:
        m = re.search(r"Stage\s+([0-6])", prompt)
        return int(m.group(1)) if m else None

    def _extract_after(self, text: str, key: str) -> str:
        m = re.search(rf"{re.escape(key)}[：:]\s*([^\n,，。]+)", text)
        return m.group(1).strip() if m else ""


class HFLocalLLM:
    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = True,
        collect_token_stats: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("HF backend needs torch and transformers installed.") from exc

        torch_dtype = dtype
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if device_map and device_map.lower() not in {"none", "null", "false"}:
            model_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if "device_map" not in model_kwargs and torch.cuda.is_available():
            self.model.to("cuda")
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.collect_token_stats = collect_token_stats

    def generate(self, prompt: str, max_new_tokens: int = 768, temperature: float = 0.7) -> Generation:
        import torch

        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            try:
                model_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                model_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            model_input = prompt

        inputs = self.tokenizer(model_input, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = temperature > 0
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = max(temperature, 1e-5)
            generate_kwargs["top_p"] = 0.9
        if self.collect_token_stats:
            generate_kwargs["return_dict_in_generate"] = True
            generate_kwargs["output_scores"] = True
        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)
        output_ids = outputs.sequences if self.collect_token_stats else outputs
        new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        diagnostics = {}
        if self.collect_token_stats and getattr(outputs, "scores", None):
            entropies = []
            top1_probs = []
            for logits in outputs.scores:
                probs = torch.softmax(logits[0].float(), dim=-1)
                entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item() / log(2))
                entropies.append(entropy)
                top1_probs.append(float(probs.max().item()))
            if entropies:
                diagnostics = {
                    "token_entropy_mean_bits": round(sum(entropies) / len(entropies), 4),
                    "token_entropy_max_bits": round(max(entropies), 4),
                    "token_entropy_min_bits": round(min(entropies), 4),
                    "token_top1_prob_mean": round(sum(top1_probs) / len(top1_probs), 4),
                }
        return Generation(
            text=text,
            input_tokens=int(inputs["input_ids"].numel()),
            output_tokens=int(new_ids.numel()),
            diagnostics=diagnostics,
        )
