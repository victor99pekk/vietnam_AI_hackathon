"""
model_eval — LoRA Fine-Tuning (Colab GPU)

Fine-tunes Qwen2.5 using 4-bit QLoRA on CUDA GPUs (Google Colab T4/V100/A100).
Designed to run efficiently in Colab — NOT for Mac.

Usage (in Colab):
    from evaluation.model_eval.finetune import FineTuner, FineTuneConfig

    config = FineTuneConfig(
        base_model="Qwen/Qwen2.5-3B-Instruct",
        max_steps=200,
    )
    ft = FineTuner(config)
    adapter_path = ft.fine_tune(
        train_data_path=Path("kg_qa_train.jsonl"),
        adapter_name="model_b_kg",
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
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# ── Optional: Unsloth (~2x faster on Colab) ─────────────────
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
        logger.info(
            "CUDA not available — will fine-tune on CPU (slower but works). "
            "Use a small model like Qwen2.5-0.5B for manageable CPU training times."
        )


@dataclass
class FineTuneConfig:
    """Configuration for a LoRA fine-tuning run (GPU or CPU)."""

    # Default: Qwen2.5-0.5B — small enough for CPU fine-tuning
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: Path = field(default_factory=lambda: Path("output_eval/method2"))

    # ── Device ──
    device: str = "cpu"  # "cpu" | "cuda" | "auto" — default CPU to work everywhere

    # ── 4-bit quantization (GPU only — ignored on CPU) ──
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # ── LoRA ──
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # ── Training ──
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 4      # reduce to 1-2 for CPU
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2.0e-4
    max_steps: int = 200
    warmup_steps: int = 20
    logging_steps: int = 10
    save_steps: int = 100
    weight_decay: float = 0.01
    seed: int = 42
    use_gradient_checkpointing: bool = True   # GPU only — ignored on CPU


class FineTuner:
    """LoRA fine-tuner — works on GPU (CUDA) or CPU.

    GPU path: 4-bit QLoRA with bitsandbytes (fast, low VRAM).
    CPU path: Full-precision LoRA with PEFT (slower but works anywhere).
    """

    def __init__(self, config: FineTuneConfig) -> None:
        self.config = config

        # Resolve device
        if config.device == "auto":
            self._device = "cuda" if CUDA_AVAILABLE else "cpu"
        else:
            self._device = config.device

        if self._device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)
        else:
            logger.info("Device: CPU (fine-tuning will be slow — use a small model)")

    @property
    def is_cpu(self) -> bool:
        return self._device == "cpu"

    def fine_tune(
        self,
        train_data_path: Path,
        adapter_name: str,
        eval_data_path: Path | None = None,
    ) -> Path:
        """Fine-tune the base model and save the LoRA adapter."""
        adapter_dir = self.config.output_dir / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = self._load_qa_dataset(train_data_path)
        eval_dataset = self._load_qa_dataset(eval_data_path) if eval_data_path else None

        logger.info(
            "Fine-tuning '%s': %d train / %d eval examples",
            adapter_name, len(train_dataset),
            len(eval_dataset) if eval_dataset else 0,
        )

        if self.is_cpu:
            logger.info("Using CPU fine-tuning (full-precision LoRA)")
            return self._train_cpu(train_dataset, eval_dataset, adapter_dir)
        elif UNSLOTH_AVAILABLE:
            logger.info("Using Unsloth (2x faster)")
            return self._train_unsloth(train_dataset, eval_dataset, adapter_dir)
        else:
            logger.info("Using bitsandbytes + PEFT")
            return self._train_bnb(train_dataset, eval_dataset, adapter_dir)

    # ── bitsandbytes + PEFT (standard Colab path) ─────────────

    def _train_bnb(
        self,
        train_dataset: list[dict[str, str]],
        eval_dataset: list[dict[str, str]] | None,
        adapter_dir: Path,
    ) -> Path:
        """Fine-tune with bitsandbytes 4-bit QLoRA."""
        cfg = self.config
        compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )

        logger.info("Loading %s with 4-bit quantization...", cfg.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

        # Tokenize
        def format_prompt(ex: dict[str, str]) -> str:
            instruction = ex.get("instruction", ex.get("question", ""))
            response = ex.get("response", ex.get("answer", ""))
            return (
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>"
            )

        train_texts = [format_prompt(ex) for ex in train_dataset]
        eval_texts = [format_prompt(ex) for ex in eval_dataset] if eval_dataset else None

        train_enc = tokenizer(
            train_texts, truncation=True, padding=True,
            max_length=cfg.max_seq_length, return_tensors="pt",
        )
        eval_enc = tokenizer(
            eval_texts, truncation=True, padding=True,
            max_length=cfg.max_seq_length, return_tensors="pt",
        ) if eval_texts else None

        class QADataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.encodings.items()}
            def __len__(self):
                return len(self.encodings["input_ids"])

        train_ds = QADataset(train_enc)
        eval_ds = QADataset(eval_enc) if eval_enc else None

        training_args = TrainingArguments(
            output_dir=str(adapter_dir),
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_steps=cfg.max_steps,
            warmup_steps=cfg.warmup_steps,
            logging_steps=cfg.logging_steps,
            save_strategy="no",             # skip checkpoints — only save final adapter
            weight_decay=cfg.weight_decay,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            seed=cfg.seed,
            fp16=True,
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none",
            save_total_limit=1,
            gradient_checkpointing=cfg.use_gradient_checkpointing,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        logger.info("Starting training: %d steps on %s",
                     cfg.max_steps, torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU")
        trainer.train()

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info("Adapter saved → %s", adapter_dir)

        self._save_metadata(adapter_dir, len(train_dataset),
                            len(eval_dataset) if eval_dataset else 0)
        return adapter_dir

    # ── CPU Path (full-precision LoRA, no quantization) ──────

    def _train_cpu(
        self,
        train_dataset: list[dict[str, str]],
        eval_dataset: list[dict[str, str]] | None,
        adapter_dir: Path,
    ) -> Path:
        """Fine-tune on CPU with full-precision LoRA (no quantization)."""
        cfg = self.config

        logger.info("Loading %s on CPU (this may take a few minutes)...", cfg.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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

        # Tokenize
        def format_prompt(ex: dict[str, str]) -> str:
            instruction = ex.get("instruction", ex.get("question", ""))
            response = ex.get("response", ex.get("answer", ""))
            return (
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>"
            )

        train_texts = [format_prompt(ex) for ex in train_dataset]
        eval_texts = [format_prompt(ex) for ex in eval_dataset] if eval_dataset else None

        train_enc = tokenizer(
            train_texts, truncation=True, padding=True,
            max_length=cfg.max_seq_length, return_tensors="pt",
        )
        eval_enc = tokenizer(
            eval_texts, truncation=True, padding=True,
            max_length=cfg.max_seq_length, return_tensors="pt",
        ) if eval_texts else None

        class QADataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.encodings.items()}
            def __len__(self):
                return len(self.encodings["input_ids"])

        train_ds = QADataset(train_enc)
        eval_ds = QADataset(eval_enc) if eval_enc else None

        # CPU-friendly training args (no fp16, no 8bit adam, no gradient ckpt, no checkpoints)
        cpu_batch_size = min(cfg.per_device_train_batch_size, 2)
        training_args = TrainingArguments(
            output_dir=str(adapter_dir),
            per_device_train_batch_size=cpu_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_steps=cfg.max_steps,
            warmup_steps=cfg.warmup_steps,
            logging_steps=cfg.logging_steps,
            save_strategy="no",             # skip checkpoints — only save final adapter
            weight_decay=cfg.weight_decay,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            seed=cfg.seed,
            fp16=False,
            bf16=False,
            report_to="none",
            save_total_limit=1,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        logger.info("Starting CPU training: %d steps", cfg.max_steps)
        trainer.train()

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info("Adapter saved → %s", adapter_dir)

        self._save_metadata(adapter_dir, len(train_dataset),
                            len(eval_dataset) if eval_dataset else 0)
        return adapter_dir

    # ── Unsloth Path (optional, ~2x faster on Colab) ──────────

    def _train_unsloth(
        self,
        train_dataset: list[dict[str, str]],
        eval_dataset: list[dict[str, str]] | None,
        adapter_dir: Path,
    ) -> Path:
        """Fine-tune using Unsloth — faster, same memory footprint."""
        cfg = self.config

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=True,
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

        def format_prompt(ex: dict[str, str]) -> str:
            instruction = ex.get("instruction", ex.get("question", ""))
            response = ex.get("response", ex.get("answer", ""))
            return (
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>"
            )

        train_texts = [format_prompt(ex) for ex in train_dataset]
        eval_texts = [format_prompt(ex) for ex in eval_dataset] if eval_dataset else None

        train_enc = tokenizer(
            train_texts, truncation=True, padding=True,
            max_length=cfg.max_seq_length, return_tensors="pt",
        )
        eval_enc = tokenizer(
            eval_texts, truncation=True, padding=True,
            max_length=cfg.max_seq_length, return_tensors="pt",
        ) if eval_texts else None

        class QADataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.encodings.items()}
            def __len__(self):
                return len(self.encodings["input_ids"])

        train_ds = QADataset(train_enc)
        eval_ds = QADataset(eval_enc) if eval_enc else None

        training_args = TrainingArguments(
            output_dir=str(adapter_dir),
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_steps=cfg.max_steps,
            warmup_steps=cfg.warmup_steps,
            logging_steps=cfg.logging_steps,
            save_strategy="no",             # skip checkpoints — only save final adapter
            weight_decay=cfg.weight_decay,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            seed=cfg.seed,
            fp16=True,
            bf16=torch.cuda.is_bf16_supported(),
            report_to="none",
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        logger.info("Starting Unsloth training: %d steps on %s",
                     cfg.max_steps, torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU")
        trainer.train()

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info("Adapter saved → %s", adapter_dir)

        self._save_metadata(adapter_dir, len(train_dataset),
                            len(eval_dataset) if eval_dataset else 0)
        return adapter_dir

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _load_qa_dataset(path: Path) -> list[dict[str, str]]:
        dataset = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if "question" in item and "instruction" not in item:
                        item["instruction"] = item["question"]
                    if "answer" in item and "response" not in item:
                        item["response"] = item["answer"]
                    dataset.append(item)
        return dataset

    def _save_metadata(self, adapter_dir: Path, n_train: int, n_eval: int) -> None:
        metadata = {
            "base_model": self.config.base_model,
            "train_samples": n_train,
            "eval_samples": n_eval,
            "max_steps": self.config.max_steps,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "load_in_4bit": self.config.load_in_4bit,
            "gpu": torch.cuda.get_device_name(0) if self._device == "cuda" else self._device,
        }
        with open(adapter_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
