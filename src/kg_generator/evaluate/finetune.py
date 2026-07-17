"""
Method 2, Step 2 — LoRA Fine-Tuning with Unsloth

Fine-tunes a lightweight Qwen2.5 model using QLoRA (4-bit quantized LoRA)
on Mac-compatible hardware (Apple MPS / Metal).

Supports two parallel training runs with identical hyperparameters:
  - Model B: KG-structured QA pairs
  - Model C: Raw-text QA pairs

Uses Unsloth for optimized training + PEFT for LoRA adapters.
Gracefully degrades if Unsloth is not available (falls back to standard
transformers + PEFT + bitsandbytes).
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Feature detection ────────────────────────────────────────

# Check for Apple Silicon
IS_APPLE_SILICON = sys.platform == "darwin" and (
    os.uname().machine == "arm64" or "Apple" in os.uname().machine
)

# Try Unsloth first, fall back to standard transformers
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    logger.info("Using Unsloth for optimized fine-tuning")
except ImportError:
    UNSLOTH_AVAILABLE = False
    logger.warning(
        "Unsloth not installed — falling back to standard transformers + PEFT. "
        "Install with: pip install unsloth"
    )

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers/peft not available — fine-tuning will be disabled")


@dataclass
class FineTuneConfig:
    """Configuration for a fine-tuning run."""
    base_model: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    output_dir: Path = field(default_factory=lambda: Path("output_eval/method2"))

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    max_steps: int = 300
    warmup_steps: int = 30
    logging_steps: int = 10
    save_steps: int = 100
    weight_decay: float = 0.01
    seed: int = 42


class FineTuner:
    """Orchestrates LoRA fine-tuning of a Qwen2.5 model on QA pairs."""

    def __init__(self, config: FineTuneConfig) -> None:
        self.config = config

    def fine_tune(
        self,
        train_data_path: Path,
        adapter_name: str,
        eval_data_path: Path | None = None,
    ) -> Path:
        """Fine-tune the base model on QA pairs and save the LoRA adapter.

        Args:
            train_data_path: Path to training JSONL file with instruction/response pairs.
            adapter_name: Name for the saved adapter (e.g., "model_b_kg").
            eval_data_path: Optional path to evaluation JSONL file.

        Returns:
            Path to the saved LoRA adapter directory.
        """
        if not TRANSFORMERS_AVAILABLE and not UNSLOTH_AVAILABLE:
            raise RuntimeError(
                "Neither Unsloth nor transformers+peft are available. "
                "Install with: pip install unsloth transformers peft accelerate bitsandbytes"
            )

        adapter_dir = self.config.output_dir / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Load and format the training data
        train_dataset = self._load_qa_dataset(train_data_path)
        eval_dataset = self._load_qa_dataset(eval_data_path) if eval_data_path else None

        logger.info(
            "Fine-tuning '%s': %d training examples, %d eval examples",
            adapter_name, len(train_dataset),
            len(eval_dataset) if eval_dataset else 0,
        )

        if UNSLOTH_AVAILABLE:
            return self._fine_tune_unsloth(train_dataset, eval_dataset, adapter_dir)
        else:
            return self._fine_tune_transformers(train_dataset, eval_dataset, adapter_dir)

    # ── Unsloth Path (recommended for Mac) ────────────────────

    def _fine_tune_unsloth(
        self,
        train_dataset: list[dict[str, str]],
        eval_dataset: list[dict[str, str]] | None,
        adapter_dir: Path,
    ) -> Path:
        """Fine-tune using Unsloth (optimized for speed + memory)."""
        from unsloth import FastLanguageModel
        import torch

        logger.info("Loading base model with Unsloth: %s", self.config.base_model)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
            dtype=None,  # Auto-detect
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
        )

        # Format datasets into the instruction format
        def format_prompt(example: dict[str, str]) -> str:
            instruction = example.get("instruction", example.get("question", ""))
            response = example.get("response", example.get("answer", ""))
            # Qwen chat format
            return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

        train_texts = [format_prompt(ex) for ex in train_dataset]
        eval_texts = [format_prompt(ex) for ex in eval_dataset] if eval_dataset else None

        # Tokenize
        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )
        eval_encodings = tokenizer(
            eval_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        ) if eval_texts else None

        # Create torch datasets
        class QADataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}

            def __len__(self):
                return len(self.encodings["input_ids"])

        train_ds = QADataset(train_encodings)
        eval_ds = QADataset(eval_encodings) if eval_encodings else None

        # Training arguments
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

        training_args = TrainingArguments(
            output_dir=str(adapter_dir),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            seed=self.config.seed,
            fp16=False,  # Mac MPS doesn't support fp16
            bf16=False,
            report_to="none",
            save_total_limit=2,
            load_best_model_at_end=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
        )

        logger.info("Starting Unsloth training for %d steps...", self.config.max_steps)
        trainer.train()

        # Save LoRA adapter
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info("LoRA adapter saved → %s", adapter_dir)

        # Save training metadata
        metadata = {
            "base_model": self.config.base_model,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset) if eval_dataset else 0,
            "max_steps": self.config.max_steps,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
        }
        with open(adapter_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return adapter_dir

    # ── Standard Transformers + PEFT Path (fallback) ──────────

    def _fine_tune_transformers(
        self,
        train_dataset: list[dict[str, str]],
        eval_dataset: list[dict[str, str]] | None,
        adapter_dir: Path,
    ) -> Path:
        """Fine-tune using standard transformers + PEFT (fallback when Unsloth unavailable)."""
        import torch

        # Determine device
        if IS_APPLE_SILICON:
            device = "mps"
            logger.info("Using Apple MPS (Metal) for training")
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            logger.warning("No GPU detected — training on CPU (will be slow!)")

        # Map Unsloth model name to standard HuggingFace name
        hf_model_name = self.config.base_model.replace("unsloth/", "").replace("-bnb-4bit", "")
        logger.info("Loading base model: %s on %s", hf_model_name, device)

        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        if device == "mps":
            model = model.to(device)

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Format datasets
        def format_prompt(example: dict[str, str]) -> str:
            instruction = example.get("instruction", example.get("question", ""))
            response = example.get("response", example.get("answer", ""))
            return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

        train_texts = [format_prompt(ex) for ex in train_dataset]

        train_encodings = tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )

        class QADataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}

            def __len__(self):
                return len(self.encodings["input_ids"])

        train_ds = QADataset(train_encodings)
        eval_ds = None

        training_args = TrainingArguments(
            output_dir=str(adapter_dir),
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            seed=self.config.seed,
            fp16=(device == "cuda"),
            bf16=False,
            report_to="none",
            save_total_limit=2,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=data_collator,
        )

        logger.info("Starting transformers training for %d steps on %s...",
                     self.config.max_steps, device)
        trainer.train()

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        logger.info("LoRA adapter saved → %s", adapter_dir)

        return adapter_dir

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _load_qa_dataset(path: Path) -> list[dict[str, str]]:
        """Load QA pairs from a JSONL file.

        Supports both the QA format (question/answer) and
        the SFT format (instruction/response).
        """
        dataset = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Normalize to {instruction, response}
                    if "question" in item and "instruction" not in item:
                        item["instruction"] = item["question"]
                    if "answer" in item and "response" not in item:
                        item["response"] = item["answer"]
                    dataset.append(item)
        return dataset
