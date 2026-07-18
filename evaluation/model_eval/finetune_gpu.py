"""
GPU-Optimized LoRA Fine-Tuning (NVIDIA H200 / High-VRAM GPUs)

Fine-tunes larger Qwen2.5 models (7B+) using mixed-precision training on
high-VRAM GPUs like H200 (141GB). Optimizations include:
  - Mixed precision (fp16/bf16) instead of 4-bit quantization (faster)
  - Larger batch sizes (8-16) for better convergence
  - Gradient checkpointing for memory efficiency with large models
  - Flash Attention v2 for faster inference
  - Higher learning rates suitable for larger models

Compared to CPU fine-tuning:
  - Uses 7B-class models instead of 0.5B
  - 5-10x faster training
  - Better convergence with larger batch sizes
  - Full-precision inference for better accuracy

Usage:
    from evaluation.model_eval.finetune_gpu import GPUFineTuner, GPUFineTuneConfig

    config = GPUFineTuneConfig(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        per_device_train_batch_size=16,
        num_train_epochs=3.0,
    )
    ft = GPUFineTuner(config)
    adapter_path = ft.fine_tune(
        train_data_path=Path("kg_qa_train.jsonl"),
        adapter_name="model_b_kg_gpu",
    )
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# ── Unsloth optimization (optional but recommended) ──────────
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

# ── CUDA check ───────────────────────────────────────────────
CUDA_AVAILABLE = False
if _TORCH_AVAILABLE:
    CUDA_AVAILABLE = torch.cuda.is_available()
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            "GPU fine-tuning requires CUDA. "
            "CUDA not detected. Use standard FineTuner for CPU training."
        )


@dataclass
class GPUFineTuneConfig:
    """Configuration for GPU-optimized LoRA fine-tuning.

    Designed for high-VRAM GPUs (H200: 141GB, A100: 80GB, etc.).
    Uses mixed precision (fp16/bf16) and larger batch sizes.
    """

    # Default: Qwen2.5-7B — use 3B or 1.5B for GPUs with <40GB
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: Path = field(default_factory=lambda: Path("output_eval/method2_gpu"))

    # ── GPU-specific: Mixed Precision (not 4-bit) ──
    use_bf16: bool = True  # Use bfloat16 if supported, else float16
    use_gradient_checkpointing: bool = True

    # ── LoRA ──
    lora_r: int = 32  # Larger rank for bigger models
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # ── Training (GPU-optimized) ──
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 16  # GPU can handle larger batches
    gradient_accumulation_steps: int = 1  # No need with large batch size
    learning_rate: float = 1.0e-4  # Smaller for stable training of large models
    max_steps: int = -1
    num_train_epochs: float = 3.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 100
    weight_decay: float = 0.01
    seed: int = 42


def _format_chat(tokenizer, instruction: str, response: str | None = None) -> str:
    """Format instruction/response as Qwen chat template."""
    messages = [{"role": "user", "content": instruction}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=response is None,
        )
    prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    return prompt if response is None else f"{prompt}{response}<|im_end|>"


def _prepare_qa_dataset(tokenizer, examples: list[dict[str, str]], max_length: int):
    """Tokenize QA records and mask user/system tokens from loss."""
    records: list[dict[str, list[int]]] = []
    for example in examples:
        instruction = example.get("instruction", example.get("question", ""))
        response = example.get("response", example.get("answer", ""))
        prompt_text = _format_chat(tokenizer, instruction)
        full_text = _format_chat(tokenizer, instruction, response)
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]
        labels = list(encoded["input_ids"])
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        if not labels or all(label == -100 for label in labels):
            continue
        records.append({
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
            "labels": labels,
        })

    if not records:
        raise ValueError("No trainable assistant tokens remained after tokenization")

    class QADataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            return records[index]

    return QADataset()


class _AssistantOnlyCollator:
    """Collate with attention to masking assistant-only loss."""
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [feature["labels"] for feature in features]
        inputs = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        padded_labels = [label + [-100] * (max_length - len(label)) for label in labels]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


class GPUFineTuner:
    """LoRA fine-tuner optimized for high-VRAM GPUs (H200, A100, etc.).

    Uses mixed-precision training (fp16/bf16) with larger batch sizes.
    Supports optional Unsloth acceleration (~2x faster).
    """

    def __init__(self, config: GPUFineTuneConfig) -> None:
        self.config = config

        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "GPU fine-tuning requires CUDA. "
                "CUDA not detected. Use standard FineTuner for CPU training."
            )

        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        gpu_count = torch.cuda.device_count()
        logger.info(
            "GPU Setup: %s (%.1f GB) — %d GPU(s) available",
            gpu_name, gpu_mem, gpu_count
        )

        # Recommend bf16 if supported (H200, modern A100)
        if config.use_bf16:
            if torch.cuda.is_bf16_supported():
                logger.info("Using bfloat16 precision (native support)")
            else:
                logger.warning(
                    "bfloat16 not supported — falling back to float16. "
                    "(Update CUDA/PyTorch for bf16 on older GPUs)"
                )

    def fine_tune(
        self,
        train_data_path: Path,
        adapter_name: str,
        eval_data_path: Path | None = None,
    ) -> Path:
        """Fine-tune base model on GPU and save LoRA adapter."""
        adapter_dir = self.config.output_dir / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = self._load_qa_dataset(train_data_path)
        eval_dataset = self._load_qa_dataset(eval_data_path) if eval_data_path else None

        logger.info(
            "GPU Fine-tuning '%s': %d train / %d eval examples",
            adapter_name, len(train_dataset),
            len(eval_dataset) if eval_dataset else 0,
        )

        if UNSLOTH_AVAILABLE:
            logger.info("Using Unsloth for acceleration (~2x faster)")
            return self._train_unsloth(train_dataset, eval_dataset, adapter_dir)
        else:
            logger.info("Using standard transformers + PEFT")
            return self._train_standard(train_dataset, eval_dataset, adapter_dir)

    def _train_standard(
        self,
        train_dataset: list[dict[str, str]],
        eval_dataset: list[dict[str, str]] | None,
        adapter_dir: Path,
    ) -> Path:
        """Fine-tune with standard transformers + mixed precision."""
        cfg = self.config

        logger.info("Loading %s with mixed precision...", cfg.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # Use Flash Attention if available
        )

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Configure LoRA
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Prepare datasets
        train_ds = _prepare_qa_dataset(tokenizer, train_dataset, cfg.max_seq_length)
        eval_ds = (
            _prepare_qa_dataset(tokenizer, eval_dataset, cfg.max_seq_length)
            if eval_dataset else None
        )

        use_bf16 = torch.cuda.is_bf16_supported() and cfg.use_bf16

        training_args = TrainingArguments(
            output_dir=str(adapter_dir),
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_steps=cfg.max_steps,
            num_train_epochs=cfg.num_train_epochs,
            warmup_steps=cfg.warmup_steps,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_ds else "no",
            weight_decay=cfg.weight_decay,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            seed=cfg.seed,
            fp16=not use_bf16,
            bf16=use_bf16,
            report_to="none",
            save_total_limit=1,
            gradient_checkpointing=cfg.use_gradient_checkpointing,
            load_best_model_at_end=bool(eval_ds),
            metric_for_best_model="eval_loss" if eval_ds else None,
            greater_is_better=False if eval_ds else None,
            ddp_find_unused_parameters=False,  # Enable DDP efficiency
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=_AssistantOnlyCollator(tokenizer),
        )

        logger.info(
            "Starting GPU training: %.1f epochs (max_steps=%d, "
            "batch_size=%d, precision=%s) on %s",
            cfg.num_train_epochs, cfg.max_steps,
            cfg.per_device_train_batch_size,
            "bf16" if use_bf16 else "fp16",
            torch.cuda.get_device_name(0),
        )
        trainer.train()

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info("GPU-trained adapter saved → %s", adapter_dir)

        self._save_metadata(adapter_dir, len(train_dataset),
                            len(eval_dataset) if eval_dataset else 0)
        return adapter_dir

    def _train_unsloth(
        self,
        train_dataset: list[dict[str, str]],
        eval_dataset: list[dict[str, str]] | None,
        adapter_dir: Path,
    ) -> Path:
        """Fine-tune using Unsloth for ~2x speedup."""
        cfg = self.config

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=False,  # Use native precision, not 4-bit
            dtype="auto",
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            target_modules=list(cfg.lora_target_modules),
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        train_ds = _prepare_qa_dataset(tokenizer, train_dataset, cfg.max_seq_length)
        eval_ds = (
            _prepare_qa_dataset(tokenizer, eval_dataset, cfg.max_seq_length)
            if eval_dataset else None
        )

        use_bf16 = torch.cuda.is_bf16_supported() and cfg.use_bf16

        training_args = TrainingArguments(
            output_dir=str(adapter_dir),
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_steps=cfg.max_steps,
            num_train_epochs=cfg.num_train_epochs,
            warmup_steps=cfg.warmup_steps,
            warmup_ratio=cfg.warmup_ratio,
            logging_steps=cfg.logging_steps,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_ds else "no",
            weight_decay=cfg.weight_decay,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            seed=cfg.seed,
            fp16=not use_bf16,
            bf16=use_bf16,
            report_to="none",
            save_total_limit=1,
            load_best_model_at_end=bool(eval_ds),
            metric_for_best_model="eval_loss" if eval_ds else None,
            greater_is_better=False if eval_ds else None,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=_AssistantOnlyCollator(tokenizer),
        )

        logger.info(
            "Starting Unsloth training: %.1f epochs (max_steps=%d, batch_size=%d) on %s",
            cfg.num_train_epochs, cfg.max_steps,
            cfg.per_device_train_batch_size,
            torch.cuda.get_device_name(0),
        )
        trainer.train()

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info("Unsloth-trained adapter saved → %s", adapter_dir)

        self._save_metadata(adapter_dir, len(train_dataset),
                            len(eval_dataset) if eval_dataset else 0)
        return adapter_dir

    def _load_qa_dataset(self, path: Path) -> list[dict[str, str]]:
        """Load QA pairs from JSONL file."""
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _save_metadata(self, adapter_dir: Path, train_count: int, eval_count: int) -> None:
        """Save training metadata for reference."""
        metadata = {
            "base_model": self.config.base_model,
            "train_samples": train_count,
            "eval_samples": eval_count,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_batch_size": self.config.per_device_train_batch_size,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "learning_rate": self.config.learning_rate,
            "device": "gpu",
            "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
        }
        with open(adapter_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
