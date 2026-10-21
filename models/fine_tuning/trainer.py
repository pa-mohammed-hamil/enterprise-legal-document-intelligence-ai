"""
Enterprise PEFT/LoRA training orchestration.

Responsibilities
----------------
- Configure Hugging Face TrainingArguments.
- Apply LoRA / QLoRA through the project's PEFT configuration.
- Train PEFT adapters.
- Evaluate and save checkpoints.
- Track training metrics.
- Persist experiment metadata.
- Support resumable training.
- Keep dataset construction in dataset_builder.py.
- Keep model loading in model_loader.py.
- Keep LoRA implementation in lora.py.

Typical flow:

    dataset_builder
          |
          v
      tokenizer
          |
          v
      base model
          |
          v
    PEFT / LoRA
          |
          v
       Trainer
          |
          v
     checkpoint
          |
          v
      evaluation
          |
          v
    model registry
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

try:
    from datasets import Dataset, DatasetDict
except ImportError:  # pragma: no cover
    Dataset = Any
    DatasetDict = Any

try:
    from transformers import (
        DataCollatorForLanguageModeling,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        Trainer,
        TrainingArguments,
    )
except ImportError:  # pragma: no cover
    DataCollatorForLanguageModeling = None
    DataCollatorWithPadding = None
    EarlyStoppingCallback = None
    PreTrainedModel = Any
    PreTrainedTokenizerBase = Any
    Trainer = None
    TrainingArguments = None

from backend.models.fine_tuning.lora import (
    LoRAConfig,
    apply_lora,
    apply_qlora,
    get_trainable_parameter_stats,
    validate_lora_model,
)
from backend.models.fine_tuning.peft_config import (
    PEFTConfig,
    PEFTMethod,
    config_to_dict,
    validate_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------


def _require_transformers() -> None:
    if Trainer is None or TrainingArguments is None:
        raise ImportError(
            "transformers is required for fine-tuning. "
            "Install it with: pip install transformers"
        )


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for fine-tuning."
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """
    Training configuration.

    Values are deliberately conservative so the configuration can be
    used as a starting point for real experiments rather than being
    tied to one particular GPU.
    """

    output_dir: str = "artifacts/fine_tuning"

    run_name: str = "legal-peft-training"

    # Training
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2

    gradient_accumulation_steps: int = 8

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05

    max_grad_norm: float = 1.0

    # Scheduler / optimizer
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"

    # Precision
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True

    # Logging
    logging_steps: int = 10
    logging_first_step: bool = True

    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 100

    # Saving
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3

    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Checkpointing
    gradient_checkpointing: bool = True

    # Reproducibility
    seed: int = 42
    data_seed: int = 42

    # Dataset behavior
    remove_unused_columns: bool = False

    # Distributed training
    ddp_find_unused_parameters: bool = False

    # Evaluation behavior
    prediction_loss_only: bool = False

    # Reporting
    report_to: tuple[str, ...] = ()

    # Early stopping
    early_stopping_patience: int | None = 3

    # Safety / limits
    max_steps: int = -1

    def __post_init__(self) -> None:

        if self.num_train_epochs <= 0:
            raise ValueError(
                "num_train_epochs must be greater than zero."
            )

        if self.per_device_train_batch_size <= 0:
            raise ValueError(
                "Training batch size must be greater than zero."
            )

        if self.per_device_eval_batch_size <= 0:
            raise ValueError(
                "Evaluation batch size must be greater than zero."
            )

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be greater than zero."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if not 0 <= self.warmup_ratio < 1:
            raise ValueError(
                "warmup_ratio must be in [0, 1)."
            )

        if self.save_total_limit <= 0:
            raise ValueError(
                "save_total_limit must be greater than zero."
            )

        if self.seed < 0:
            raise ValueError(
                "seed cannot be negative."
            )


@dataclass
class TrainingResult:
    """
    Serializable result of a training run.
    """

    run_name: str

    output_dir: str

    status: str

    started_at: float
    completed_at: float | None = None

    trainable_parameters: int = 0
    total_parameters: int = 0
    trainable_percentage: float = 0.0

    train_loss: float | None = None
    eval_loss: float | None = None

    perplexity: float | None = None

    global_step: int = 0

    best_checkpoint: str | None = None

    metrics: dict[str, Any] = field(
        default_factory=dict
    )

    error: str | None = None


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(
    seed: int,
) -> None:
    """
    Set deterministic seeds where practical.

    Full bit-for-bit reproducibility is not guaranteed across different
    hardware, CUDA versions, distributed configurations, or kernels.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device / precision helpers
# ---------------------------------------------------------------------------


def detect_precision(
    config: TrainerConfig,
) -> tuple[bool, bool]:
    """
    Resolve fp16/bf16 settings against hardware capability.
    """

    if not torch.cuda.is_available():
        return False, False

    bf16_supported = (
        hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )

    use_bf16 = (
        config.bf16
        and bf16_supported
    )

    use_fp16 = (
        config.fp16
        and not use_bf16
    )

    return use_fp16, use_bf16


# ---------------------------------------------------------------------------
# Data collators
# ---------------------------------------------------------------------------


def create_causal_lm_collator(
    tokenizer: PreTrainedTokenizerBase,
):
    """
    Create a collator for causal language-model training.
    """

    if DataCollatorForLanguageModeling is None:
        raise ImportError(
            "transformers is required."
        )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer requires either pad_token or eos_token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )


def create_classification_collator(
    tokenizer: PreTrainedTokenizerBase,
):
    """
    Create a padding collator for classification/token classification.
    """

    if DataCollatorWithPadding is None:
        raise ImportError(
            "transformers is required."
        )

    return DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize_text_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "text",
    max_length: int = 2048,
    truncation: bool = True,
    padding: bool | str = False,
) -> Dataset:
    """
    Tokenize a simple text dataset.
    """

    if text_column not in dataset.column_names:
        raise ValueError(
            f"Dataset does not contain '{text_column}'. "
            f"Available columns: {dataset.column_names}"
        )

    if max_length <= 0:
        raise ValueError(
            "max_length must be greater than zero."
        )

    def tokenize_batch(
        batch: Mapping[str, Sequence[Any]],
    ) -> Mapping[str, Any]:

        texts = batch[text_column]

        return tokenizer(
            list(texts),
            truncation=truncation,
            max_length=max_length,
            padding=padding,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        desc="Tokenizing dataset",
    )


# ---------------------------------------------------------------------------
# PEFT model preparation
# ---------------------------------------------------------------------------


def build_peft_model(
    model,
    peft_config: PEFTConfig,
):
    """
    Apply the requested PEFT method to a base model.
    """

    validate_config(peft_config)

    if peft_config.method == PEFTMethod.QLORA:

        lora_config = LoRAConfig(
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            bias=peft_config.bias,
            target_modules=list(
                peft_config.target_modules
            ),
            modules_to_save=(
                list(peft_config.modules_to_save)
                if peft_config.modules_to_save
                else None
            ),
            use_rslora=peft_config.use_rslora,
            use_dora=peft_config.use_dora,
            task_type=peft_config.task.value,
            inference_mode=peft_config.inference_mode,
        )

        peft_model = apply_qlora(
            model,
            config=lora_config,
        )

    else:

        lora_config = LoRAConfig(
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            bias=peft_config.bias,
            target_modules=list(
                peft_config.target_modules
            ),
            modules_to_save=(
                list(peft_config.modules_to_save)
                if peft_config.modules_to_save
                else None
            ),
            use_rslora=peft_config.use_rslora,
            use_dora=peft_config.use_dora,
            task_type=peft_config.task.value,
            inference_mode=peft_config.inference_mode,
        )

        peft_model = apply_lora(
            model,
            config=lora_config,
        )

    validate_lora_model(
        peft_model
    )

    return peft_model


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------


def create_training_arguments(
    config: TrainerConfig,
) -> TrainingArguments:
    """
    Convert project training configuration into Hugging Face
    TrainingArguments.
    """

    _require_transformers()

    fp16, bf16 = detect_precision(
        config
    )

    # Some environments cannot use the bitsandbytes optimizer.
    # The caller can explicitly override config.optim.
    return TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.run_name,

        num_train_epochs=config.num_train_epochs,

        per_device_train_batch_size=(
            config.per_device_train_batch_size
        ),
        per_device_eval_batch_size=(
            config.per_device_eval_batch_size
        ),

        gradient_accumulation_steps=(
            config.gradient_accumulation_steps
        ),

        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,

        max_grad_norm=config.max_grad_norm,

        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,

        fp16=fp16,
        bf16=bf16,
        tf32=(
            config.tf32
            and torch.cuda.is_available()
        ),

        logging_strategy="steps",
        logging_steps=config.logging_steps,
        logging_first_step=config.logging_first_step,

        eval_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,

        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,

        load_best_model_at_end=(
            config.load_best_model_at_end
        ),
        metric_for_best_model=(
            config.metric_for_best_model
        ),
        greater_is_better=config.greater_is_better,

        gradient_checkpointing=(
            config.gradient_checkpointing
        ),

        seed=config.seed,
        data_seed=config.data_seed,

        remove_unused_columns=(
            config.remove_unused_columns
        ),

        ddp_find_unused_parameters=(
            config.ddp_find_unused_parameters
        ),

        prediction_loss_only=(
            config.prediction_loss_only
        ),

        report_to=list(config.report_to),

        max_steps=config.max_steps,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_perplexity(
    loss: float | None,
) -> float | None:
    """
    Convert language-model loss to perplexity.

    Returns None when loss is invalid or numerically unsafe.
    """

    if loss is None:
        return None

    if not math.isfinite(loss):
        return None

    try:
        perplexity = math.exp(loss)
    except OverflowError:
        return None

    if not math.isfinite(perplexity):
        return None

    return perplexity


def default_compute_metrics(
    evaluation_result,
) -> dict[str, float]:
    """
    Default metric callback for language-model evaluation.

    Hugging Face Trainer primarily reports loss for causal LM.
    """

    metrics = {}

    if hasattr(
        evaluation_result,
        "metrics",
    ):
        raw_metrics = evaluation_result.metrics

        eval_loss = raw_metrics.get(
            "eval_loss"
        )

        perplexity = compute_perplexity(
            eval_loss
        )

        if perplexity is not None:
            metrics["perplexity"] = perplexity

    return metrics


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------


def create_trainer(
    *,
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    training_config: TrainerConfig,
    data_collator=None,
    compute_metrics: Callable | None = None,
):
    """
    Create a Hugging Face Trainer.
    """

    _require_transformers()

    arguments = create_training_arguments(
        training_config
    )

    callbacks = []

    if (
        training_config.early_stopping_patience
        is not None
        and EarlyStoppingCallback is not None
        and eval_dataset is not None
    ):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=(
                    training_config.early_stopping_patience
                )
            )
        )

    return Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def save_training_metadata(
    *,
    output_dir: str,
    training_config: TrainerConfig,
    peft_config: PEFTConfig,
    result: TrainingResult,
) -> Path:
    """
    Persist reproducible training metadata.
    """

    directory = Path(
        output_dir
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        directory / "training_metadata.json"
    )

    payload = {
        "training": asdict(
            training_config
        ),
        "peft": config_to_dict(
            peft_config
        ),
        "result": asdict(
            result
        ),
    }

    metadata_path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return metadata_path


# ---------------------------------------------------------------------------
# Training service
# ---------------------------------------------------------------------------


class PEFTTrainer:
    """
    High-level PEFT training service.

    This class intentionally does not load models or construct datasets.
    Those responsibilities remain in their respective project modules.
    """

    def __init__(
        self,
        *,
        training_config: TrainerConfig | None = None,
    ) -> None:

        self.training_config = (
            training_config
            or TrainerConfig()
        )

    def train(
        self,
        *,
        model,
        tokenizer,
        peft_config: PEFTConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        data_collator=None,
        compute_metrics: Callable | None = None,
        resume_from_checkpoint: str | bool | None = None,
    ) -> TrainingResult:
        """
        Execute a PEFT training run.
        """

        _require_transformers()
        _require_torch()

        set_seed(
            self.training_config.seed
        )

        started_at = time.time()

        result = TrainingResult(
            run_name=self.training_config.run_name,
            output_dir=self.training_config.output_dir,
            status="running",
            started_at=started_at,
        )

        try:

            logger.info(
                "Starting PEFT training: %s",
                self.training_config.run_name,
            )

            logger.info(
                "PEFT configuration: %s",
                peft_config.name,
            )

            # ----------------------------------------------------------
            # Prepare model
            # ----------------------------------------------------------

            model = build_peft_model(
                model=model,
                peft_config=peft_config,
            )

            stats = get_trainable_parameter_stats(
                model
            )

            result.trainable_parameters = (
                stats.trainable_parameters
            )

            result.total_parameters = (
                stats.total_parameters
            )

            result.trainable_percentage = (
                stats.trainable_percentage
            )

            logger.info(
                "Trainable parameters: %s / %s (%.4f%%)",
                f"{stats.trainable_parameters:,}",
                f"{stats.total_parameters:,}",
                stats.trainable_percentage,
            )

            # ----------------------------------------------------------
            # Default collator
            # ----------------------------------------------------------

            if data_collator is None:

                data_collator = (
                    create_causal_lm_collator(
                        tokenizer
                    )
                )

            # ----------------------------------------------------------
            # Trainer
            # ----------------------------------------------------------

            trainer = create_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=(
                    self.training_config
                ),
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            # ----------------------------------------------------------
            # Training
            # ----------------------------------------------------------

            train_output = trainer.train(
                resume_from_checkpoint=(
                    resume_from_checkpoint
                )
            )

            result.global_step = int(
                getattr(
                    trainer.state,
                    "global_step",
                    0,
                )
            )

            if train_output.metrics:
                result.metrics.update(
                    train_output.metrics
                )

                result.train_loss = (
                    train_output.metrics.get(
                        "train_loss"
                    )
                )

            # ----------------------------------------------------------
            # Evaluation
            # ----------------------------------------------------------

            if eval_dataset is not None:

                evaluation_metrics = (
                    trainer.evaluate()
                )

                result.metrics.update(
                    evaluation_metrics
                )

                result.eval_loss = (
                    evaluation_metrics.get(
                        "eval_loss"
                    )
                )

                result.perplexity = (
                    compute_perplexity(
                        result.eval_loss
                    )
                )

            # ----------------------------------------------------------
            # Save adapter
            # ----------------------------------------------------------

            trainer.save_model(
                self.training_config.output_dir
            )

            if tokenizer is not None:
                tokenizer.save_pretrained(
                    self.training_config.output_dir
                )

            # ----------------------------------------------------------
            # Best checkpoint
            # ----------------------------------------------------------

            result.best_checkpoint = getattr(
                trainer.state,
                "best_model_checkpoint",
                None,
            )

            result.status = "completed"
            result.completed_at = time.time()

            save_training_metadata(
                output_dir=(
                    self.training_config.output_dir
                ),
                training_config=(
                    self.training_config
                ),
                peft_config=peft_config,
                result=result,
            )

            logger.info(
                "PEFT training completed successfully."
            )

            return result

        except Exception as exc:

            result.status = "failed"
            result.completed_at = time.time()
            result.error = (
                f"{type(exc).__name__}: {exc}"
            )

            try:
                save_training_metadata(
                    output_dir=(
                        self.training_config.output_dir
                    ),
                    training_config=(
                        self.training_config
                    ),
                    peft_config=peft_config,
                    result=result,
                )
            except Exception:
                logger.exception(
                    "Failed to save failed-run metadata."
                )

            logger.exception(
                "PEFT training failed."
            )

            raise


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def train_peft_model(
    *,
    model,
    tokenizer,
    peft_config: PEFTConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    training_config: TrainerConfig | None = None,
    data_collator=None,
    compute_metrics: Callable | None = None,
    resume_from_checkpoint: str | bool | None = None,
) -> TrainingResult:
    """
    Convenience wrapper around PEFTTrainer.
    """

    trainer = PEFTTrainer(
        training_config=training_config
    )

    return trainer.train(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        resume_from_checkpoint=(
            resume_from_checkpoint
        ),
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


__all__ = [
    "TrainerConfig",
    "TrainingResult",
    "PEFTTrainer",
    "set_seed",
    "detect_precision",
    "tokenize_text_dataset",
    "create_causal_lm_collator",
    "create_classification_collator",
    "build_peft_model",
    "create_training_arguments",
    "compute_perplexity",
    "default_compute_metrics",
    "create_trainer",
    "save_training_metadata",
    "train_peft_model",
]