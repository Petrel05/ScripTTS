from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int


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
                f"种子{suffix}：主角最初以为问题来自外部系统，调查后发现真正的冲突来自被忽视的人际关系。"
                "结尾反转：所谓失控其实是在保护某个脆弱的承诺。"
            )
        if stage == 2:
            return (
                "Logline：一次看似异常的技术事件，把三个人推到必须互相信任的夜晚。\n"
                "开端：校园或工作空间中出现反常提示。\n"
                "冲突：主角怀疑系统失控，旁人坚持按规则处理。\n"
                "转折：线索显示异常与过去的一个隐瞒有关。\n"
                "高潮：主角必须公开真相或保护他人。\n"
                "结尾：系统并未失控，它只是提前执行了一个人类不敢说出口的请求。"
            )
        if stage == 3:
            return (
                "人物表：\n"
                "林舟：行动型主角，想证明自己判断正确，深层动机是被信任。\n"
                "许老师：规则维护者，担心技术破坏秩序，也害怕自己判断失误。\n"
                "小艾：AI或智能系统，表面冷静，功能是不断制造误解并保存关键证据。\n"
                "冲突关系：林舟与许老师是信任冲突；林舟与小艾是误解到合作；许老师与小艾是控制权冲突。"
            )
        if stage == 4:
            return (
                "第1场：异常提示。主角收到系统警告，旁人认为是误报，冲突启动。\n"
                "第2场：误导线索。众人发现证据指向主角，主角被迫自证。\n"
                "第3场：反向调查。主角发现系统一直在回避某个名字。\n"
                "第4场：真相公开。反转揭示系统的异常是为了保护一个被忽略的人。"
            )
        if stage == 5:
            m = re.search(r"当前场[：:]\s*(.+)", prompt)
            goal = m.group(1).strip() if m else "冲突推进"
            return (
                f"【舞台说明】{goal}。灯光收紧，屏幕亮起。\n"
                "林舟：如果这只是故障，它为什么只提醒我一个人？\n"
                "许老师：规则不是用来猜的，证据在哪里？\n"
                "小艾：证据已保存，但公开将改变你们对彼此的判断。\n"
                "【本场结尾钩子】屏幕停在一个被删除的时间戳上。"
            )
        if "最终评价" in prompt or "judge" in prompt.lower():
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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
        with torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)
        new_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return Generation(text=text, input_tokens=int(inputs["input_ids"].numel()), output_tokens=int(new_ids.numel()))
