"""
backend/models/fine_tuning/training_config.py

Central training configuration for the legal AI fine-tuning pipeline.

Responsibilities
----------------
- Training hyperparameters.
- Optimizer and scheduler configuration.
- Batch-size and gradient accumulation policy.
- Precision configuration.
- Evaluation/checkpoint configuration.
- Reproducibility settings.
- Distributed-training settings.
- Early-stopping policy.
- Resource/runtime configuration.
- Validation and serialization.

This module does NOT:
- Load models.
- Build datasets.
- Apply LoRA/QLoRA.
- Execute training.

Those responsibilities belong to:
    dataset_builder.py
    lora.py
    peft_config.py
    trainer.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================================
# ENUMS
# ============================================================================


class OptimizerType(str, Enum):
    """Supported optimizer families."""

    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_8BIT = "adamw_8bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    SGD = "sgd"


class SchedulerType(str, Enum):
    """Learning-rate scheduler strategies."""

    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    POLYNOMIAL = "polynomial"


class PrecisionType(str, Enum):
    """Training precision modes."""

    AUTO = "auto"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class EvaluationStrategy(str, Enum):
    """Evaluation scheduling."""

    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class SaveStrategy(str, Enum):
    """Checkpoint scheduling."""

    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class LoggingStrategy(str, Enum):
    """Logging scheduling."""

    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class ReportBackend(str, Enum):
    """Experiment tracking integrations."""

    NONE = "none"
    WANDB = "wandb"
    MLFLOW = "mlflow"
    TENSORBOARD = "tensorboard"


# ============================================================================
# MAIN CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class TrainingConfig:
    """
    Complete configuration for a fine-tuning experiment.

    The configuration is intentionally independent from PEFTConfig so that
    the same training policy can be used with LoRA, QLoRA, or another
    parameter-efficient method.
    """

    # ------------------------------------------------------------------
    # Experiment identity
    # ------------------------------------------------------------------

    name: str = "legal-ai-training"

    version: str = "1.0.0"

    output_dir: str = (
        "artifacts/fine_tuning/legal-ai-training"
    )

    logging_dir: str = (
        "artifacts/fine_tuning/logs"
    )

    # ------------------------------------------------------------------
    # Training duration
    # ------------------------------------------------------------------

    num_train_epochs: float = 3.0

    max_steps: int = -1

    # ------------------------------------------------------------------
    # Batch configuration
    # ------------------------------------------------------------------

    per_device_train_batch_size: int = 2

    per_device_eval_batch_size: int = 2

    gradient_accumulation_steps: int = 8

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    learning_rate: float = 2e-4

    weight_decay: float = 0.01

    max_grad_norm: float = 1.0

    optimizer: OptimizerType = (
        OptimizerType.PAGED_ADAMW_8BIT
    )

    scheduler: SchedulerType = (
        SchedulerType.COSINE
    )

    warmup_ratio: float = 0.05

    warmup_steps: int = 0

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------

    precision: PrecisionType = (
        PrecisionType.AUTO
    )

    tf32: bool = True

    # ------------------------------------------------------------------
    # Gradient behavior
    # ------------------------------------------------------------------

    gradient_checkpointing: bool = True

    gradient_checkpointing_kwargs: dict[str, Any] = (
        field(
            default_factory=dict
        )
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    evaluation_strategy: EvaluationStrategy = (
        EvaluationStrategy.STEPS
    )

    eval_steps: int = 100

    prediction_loss_only: bool = False

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    save_strategy: SaveStrategy = (
        SaveStrategy.STEPS
    )

    save_steps: int = 100

    save_total_limit: int = 3

    save_safetensors: bool = True

    load_best_model_at_end: bool = True

    metric_for_best_model: str = "eval_loss"

    greater_is_better: bool = False

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    logging_strategy: LoggingStrategy = (
        LoggingStrategy.STEPS
    )

    logging_steps: int = 10

    logging_first_step: bool = True

    report_backend: ReportBackend = (
        ReportBackend.NONE
    )

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    early_stopping_enabled: bool = True

    early_stopping_patience: int = 3

    early_stopping_threshold: float = 0.0

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    seed: int = 42

    data_seed: int = 42

    deterministic: bool = False

    # ------------------------------------------------------------------
    # Data-loader settings
    # ------------------------------------------------------------------

    dataloader_num_workers: int = 2

    dataloader_pin_memory: bool = True

    dataloader_persistent_workers: bool = False

    dataloader_drop_last: bool = False

    # ------------------------------------------------------------------
    # Dataset behavior
    # ------------------------------------------------------------------

    remove_unused_columns: bool = False

    label_smoothing_factor: float = 0.0

    # ------------------------------------------------------------------
    # Distributed training
    # ------------------------------------------------------------------

    ddp_find_unused_parameters: bool = False

    ddp_broadcast_buffers: bool = False

    # ------------------------------------------------------------------
    # Memory / performance
    # ------------------------------------------------------------------

    use_cache: bool = False

    group_by_length: bool = False

    length_column_name: str | None = None

    # ------------------------------------------------------------------
    # Hub / artifact publishing
    # ------------------------------------------------------------------

    push_to_hub: bool = False

    hub_model_id: str | None = None

    hub_private_repo: bool = True

    hub_strategy: str = "every_save"

    # ------------------------------------------------------------------
    # Runtime policy
    # ------------------------------------------------------------------

    resume_from_checkpoint: str | None = None

    overwrite_output_dir: bool = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    description: str = ""

    tags: tuple[str, ...] = field(
        default_factory=tuple
    )

    extra: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate configuration invariants."""

        if not self.name.strip():
            raise ValueError(
                "Training configuration name cannot be empty."
            )

        if not self.output_dir.strip():
            raise ValueError(
                "output_dir cannot be empty."
            )

        if self.num_train_epochs <= 0:
            raise ValueError(
                "num_train_epochs must be greater than zero."
            )

        if self.max_steps == 0:
            raise ValueError(
                "max_steps must be -1 or greater than zero."
            )

        if self.per_device_train_batch_size <= 0:
            raise ValueError(
                "per_device_train_batch_size must be greater than zero."
            )

        if self.per_device_eval_batch_size <= 0:
            raise ValueError(
                "per_device_eval_batch_size must be greater than zero."
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

        if self.max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be greater than zero."
            )

        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(
                "warmup_ratio must be in [0, 1)."
            )

        if self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps cannot be negative."
            )

        if self.eval_steps <= 0:
            raise ValueError(
                "eval_steps must be greater than zero."
            )

        if self.save_steps <= 0:
            raise ValueError(
                "save_steps must be greater than zero."
            )

        if self.logging_steps <= 0:
            raise ValueError(
                "logging_steps must be greater than zero."
            )

        if self.save_total_limit <= 0:
            raise ValueError(
                "save_total_limit must be greater than zero."
            )

        if not 0.0 <= self.label_smoothing_factor <= 1.0:
            raise ValueError(
                "label_smoothing_factor must be in [0, 1]."
            )

        if self.early_stopping_enabled:
            if self.early_stopping_patience <= 0:
                raise ValueError(
                    "early_stopping_patience must be greater than zero."
                )

            if self.early_stopping_threshold < 0:
                raise ValueError(
                    "early_stopping_threshold cannot be negative."
                )

        if self.dataloader_num_workers < 0:
            raise ValueError(
                "dataloader_num_workers cannot be negative."
            )

        if self.seed < 0:
            raise ValueError(
                "seed cannot be negative."
            )

        if self.data_seed < 0:
            raise ValueError(
                "data_seed cannot be negative."
            )

        if self.push_to_hub and not self.hub_model_id:
            raise ValueError(
                "hub_model_id is required when push_to_hub=True."
            )

        if (
            self.load_best_model_at_end
            and self.evaluation_strategy
            == EvaluationStrategy.NO
        ):
            raise ValueError(
                "load_best_model_at_end requires evaluation."
            )

        if (
            self.load_best_model_at_end
            and self.save_strategy
            == SaveStrategy.NO
        ):
            raise ValueError(
                "load_best_model_at_end requires checkpoint saving."
            )

        if (
            self.deterministic
            and self.dataloader_num_workers > 0
        ):
            # This is not technically invalid, but deterministic
            # data loading can be harder with multiple workers.
            # We intentionally leave the configuration valid.
            pass


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================


LEGAL_LLM_DEFAULT = TrainingConfig(
    name="legal-llm-default",
    output_dir=(
        "artifacts/fine_tuning/legal-llm-default"
    ),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    optimizer=OptimizerType.PAGED_ADAMW_8BIT,
    scheduler=SchedulerType.COSINE,
    warmup_ratio=0.05,
    precision=PrecisionType.AUTO,
    gradient_checkpointing=True,
    evaluation_strategy=EvaluationStrategy.STEPS,
    eval_steps=100,
    save_strategy=SaveStrategy.STEPS,
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    early_stopping_enabled=True,
    early_stopping_patience=3,
    tags=(
        "legal",
        "llm",
        "peft",
    ),
)


LEGAL_QLORA_MEMORY_EFFICIENT = TrainingConfig(
    name="legal-qlora-memory-efficient",
    output_dir=(
        "artifacts/fine_tuning/legal-qlora-memory-efficient"
    ),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    weight_decay=0.01,
    optimizer=OptimizerType.PAGED_ADAMW_8BIT,
    scheduler=SchedulerType.COSINE,
    warmup_ratio=0.05,
    precision=PrecisionType.BF16,
    tf32=True,
    gradient_checkpointing=True,
    evaluation_strategy=EvaluationStrategy.STEPS,
    eval_steps=100,
    save_strategy=SaveStrategy.STEPS,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=2,
    tags=(
        "legal",
        "qlora",
        "memory-efficient",
        "4bit",
    ),
)


CLASSIFICATION_DEFAULT = TrainingConfig(
    name="legal-classification-default",
    output_dir=(
        "artifacts/fine_tuning/legal-classification"
    ),
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    optimizer=OptimizerType.ADAMW_TORCH,
    scheduler=SchedulerType.LINEAR,
    warmup_ratio=0.1,
    precision=PrecisionType.AUTO,
    gradient_checkpointing=False,
    evaluation_strategy=EvaluationStrategy.STEPS,
    eval_steps=100,
    save_strategy=SaveStrategy.STEPS,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    early_stopping_enabled=True,
    early_stopping_patience=3,
    tags=(
        "legal",
        "classification",
        "encoder",
    ),
)


NER_DEFAULT = TrainingConfig(
    name="legal-ner-default",
    output_dir=(
        "artifacts/fine_tuning/legal-ner"
    ),
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=3e-5,
    weight_decay=0.01,
    optimizer=OptimizerType.ADAMW_TORCH,
    scheduler=SchedulerType.LINEAR,
    warmup_ratio=0.1,
    precision=PrecisionType.AUTO,
    gradient_checkpointing=False,
    evaluation_strategy=EvaluationStrategy.STEPS,
    eval_steps=100,
    save_strategy=SaveStrategy.STEPS,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    tags=(
        "legal",
        "ner",
        "token-classification",
    ),
)


# ============================================================================
# REGISTRY
# ============================================================================


TRAINING_CONFIG_REGISTRY: dict[
    str,
    TrainingConfig,
] = {
    config.name: config
    for config in (
        LEGAL_LLM_DEFAULT,
        LEGAL_QLORA_MEMORY_EFFICIENT,
        CLASSIFICATION_DEFAULT,
        NER_DEFAULT,
    )
}


# ============================================================================
# REGISTRY OPERATIONS
# ============================================================================


def register_training_config(
    config: TrainingConfig,
    *,
    overwrite: bool = False,
) -> None:
    """
    Register a custom training configuration.
    """

    config.validate()

    if (
        config.name in TRAINING_CONFIG_REGISTRY
        and not overwrite
    ):
        raise ValueError(
            f"Training configuration '{config.name}' "
            "already exists."
        )

    TRAINING_CONFIG_REGISTRY[
        config.name
    ] = config


def get_training_config(
    name: str,
) -> TrainingConfig:
    """Retrieve a registered training configuration."""

    try:
        return TRAINING_CONFIG_REGISTRY[name]
    except KeyError as exc:

        available = ", ".join(
            sorted(
                TRAINING_CONFIG_REGISTRY
            )
        )

        raise KeyError(
            f"Unknown training configuration '{name}'. "
            f"Available configurations: {available}"
        ) from exc


def list_training_configs() -> list[str]:
    """Return registered configuration names."""

    return sorted(
        TRAINING_CONFIG_REGISTRY.keys()
    )


# ============================================================================
# DEVICE / PRECISION HELPERS
# ============================================================================


def resolve_precision(
    config: TrainingConfig,
    *,
    cuda_available: bool,
    bf16_supported: bool = False,
) -> tuple[bool, bool]:
    """
    Resolve effective fp16/bf16 settings.

    Returns:
        (use_fp16, use_bf16)
    """

    if config.precision == PrecisionType.FP32:
        return False, False

    if not cuda_available:
        return False, False

    if config.precision == PrecisionType.BF16:

        if not bf16_supported:
            raise RuntimeError(
                "BF16 was explicitly requested, but the "
                "available CUDA device does not support BF16."
            )

        return False, True

    if config.precision == PrecisionType.FP16:
        return True, False

    # AUTO
    if bf16_supported:
        return False, True

    return True, False


# ============================================================================
# EFFECTIVE BATCH SIZE
# ============================================================================


def effective_batch_size(
    config: TrainingConfig,
    *,
    world_size: int = 1,
) -> int:
    """
    Calculate effective global batch size.

    Formula:

        per_device_batch
        × gradient_accumulation
        × number_of_processes
    """

    if world_size <= 0:
        raise ValueError(
            "world_size must be greater than zero."
        )

    return (
        config.per_device_train_batch_size
        * config.gradient_accumulation_steps
        * world_size
    )


# ============================================================================
# TRAINING ARGUMENT CONVERSION
# ============================================================================


def to_training_arguments_kwargs(
    config: TrainingConfig,
) -> dict[str, Any]:
    """
    Convert the project configuration into keyword arguments
    suitable for Hugging Face TrainingArguments.

    Hardware-dependent precision is intentionally resolved later.
    """

    config.validate()

    kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "logging_dir": config.logging_dir,

        "run_name": config.name,

        "num_train_epochs": (
            config.num_train_epochs
        ),
        "max_steps": config.max_steps,

        "per_device_train_batch_size": (
            config.per_device_train_batch_size
        ),
        "per_device_eval_batch_size": (
            config.per_device_eval_batch_size
        ),
        "gradient_accumulation_steps": (
            config.gradient_accumulation_steps
        ),

        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,

        "lr_scheduler_type": (
            config.scheduler.value
        ),

        "warmup_ratio": config.warmup_ratio,
        "warmup_steps": config.warmup_steps,

        "gradient_checkpointing": (
            config.gradient_checkpointing
        ),
        "gradient_checkpointing_kwargs": (
            config.gradient_checkpointing_kwargs
            or None
        ),

        "eval_strategy": (
            config.evaluation_strategy.value
        ),
        "eval_steps": config.eval_steps,

        "save_strategy": (
            config.save_strategy.value
        ),
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "save_safetensors": config.save_safetensors,

        "load_best_model_at_end": (
            config.load_best_model_at_end
        ),
        "metric_for_best_model": (
            config.metric_for_best_model
        ),
        "greater_is_better": (
            config.greater_is_better
        ),

        "logging_strategy": (
            config.logging_strategy.value
        ),
        "logging_steps": config.logging_steps,
        "logging_first_step": (
            config.logging_first_step
        ),

        "seed": config.seed,
        "data_seed": config.data_seed,

        "dataloader_num_workers": (
            config.dataloader_num_workers
        ),
        "dataloader_pin_memory": (
            config.dataloader_pin_memory
        ),
        "dataloader_persistent_workers": (
            config.dataloader_persistent_workers
        ),
        "dataloader_drop_last": (
            config.dataloader_drop_last
        ),

        "remove_unused_columns": (
            config.remove_unused_columns
        ),

        "label_smoothing_factor": (
            config.label_smoothing_factor
        ),

        "ddp_find_unused_parameters": (
            config.ddp_find_unused_parameters
        ),
        "ddp_broadcast_buffers": (
            config.ddp_broadcast_buffers
        ),

        "tf32": config.tf32,

        "prediction_loss_only": (
            config.prediction_loss_only
        ),

        "push_to_hub": config.push_to_hub,

        "hub_model_id": config.hub_model_id,

        "hub_private_repo": (
            config.hub_private_repo
        ),

        "hub_strategy": config.hub_strategy,

        "overwrite_output_dir": (
            config.overwrite_output_dir
        ),
    }

    # Hugging Face accepts "report_to" as a list.
    if config.report_backend == ReportBackend.NONE:
        kwargs["report_to"] = []
    else:
        kwargs["report_to"] = [
            config.report_backend.value
        ]

    # The optimizer is passed as a string.
    kwargs["optim"] = config.optimizer.value

    return kwargs


# ============================================================================
# SERIALIZATION
# ============================================================================


def config_to_dict(
    config: TrainingConfig,
) -> dict[str, Any]:
    """
    Serialize configuration for experiment tracking.
    """

    data = asdict(config)

    data["optimizer"] = (
        config.optimizer.value
    )

    data["scheduler"] = (
        config.scheduler.value
    )

    data["precision"] = (
        config.precision.value
    )

    data["evaluation_strategy"] = (
        config.evaluation_strategy.value
    )

    data["save_strategy"] = (
        config.save_strategy.value
    )

    data["logging_strategy"] = (
        config.logging_strategy.value
    )

    data["report_backend"] = (
        config.report_backend.value
    )

    data["tags"] = list(
        config.tags
    )

    return data


def config_from_dict(
    data: dict[str, Any],
) -> TrainingConfig:
    """
    Reconstruct TrainingConfig from serialized data.
    """

    payload = dict(data)

    payload["optimizer"] = OptimizerType(
        payload.get(
            "optimizer",
            OptimizerType.ADAMW_TORCH.value,
        )
    )

    payload["scheduler"] = SchedulerType(
        payload.get(
            "scheduler",
            SchedulerType.LINEAR.value,
        )
    )

    payload["precision"] = PrecisionType(
        payload.get(
            "precision",
            PrecisionType.AUTO.value,
        )
    )

    payload["evaluation_strategy"] = (
        EvaluationStrategy(
            payload.get(
                "evaluation_strategy",
                EvaluationStrategy.STEPS.value,
            )
        )
    )

    payload["save_strategy"] = SaveStrategy(
        payload.get(
            "save_strategy",
            SaveStrategy.STEPS.value,
        )
    )

    payload["logging_strategy"] = (
        LoggingStrategy(
            payload.get(
                "logging_strategy",
                LoggingStrategy.STEPS.value,
            )
        )
    )

    payload["report_backend"] = ReportBackend(
        payload.get(
            "report_backend",
            ReportBackend.NONE.value,
        )
    )

    payload["tags"] = tuple(
        payload.get(
            "tags",
            (),
        )
    )

    return TrainingConfig(
        **payload
    )


# ============================================================================
# FILE OPERATIONS
# ============================================================================


def save_config(
    config: TrainingConfig,
    path: str | Path,
) -> Path:
    """
    Save configuration as JSON.
    """

    import json

    config.validate()

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            config_to_dict(config),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return output_path


def load_config(
    path: str | Path,
) -> TrainingConfig:
    """
    Load TrainingConfig from JSON.
    """

    import json

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Training configuration not found: "
            f"{input_path}"
        )

    payload = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

    return config_from_dict(
        payload
    )


# ============================================================================
# SUMMARY
# ============================================================================


def describe_config(
    config: TrainingConfig,
) -> dict[str, Any]:
    """
    Return a compact experiment summary.
    """

    return {
        "name": config.name,
        "optimizer": config.optimizer.value,
        "scheduler": config.scheduler.value,
        "precision": config.precision.value,
        "learning_rate": config.learning_rate,
        "epochs": config.num_train_epochs,
        "train_batch_size": (
            config.per_device_train_batch_size
        ),
        "gradient_accumulation": (
            config.gradient_accumulation_steps
        ),
        "gradient_checkpointing": (
            config.gradient_checkpointing
        ),
        "evaluation": (
            config.evaluation_strategy.value
        ),
        "checkpointing": (
            config.save_strategy.value
        ),
        "effective_batch_size_single_gpu": (
            effective_batch_size(config)
        ),
        "tags": list(config.tags),
    }


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "OptimizerType",
    "SchedulerType",
    "PrecisionType",
    "EvaluationStrategy",
    "SaveStrategy",
    "LoggingStrategy",
    "ReportBackend",
    "TrainingConfig",
    "LEGAL_LLM_DEFAULT",
    "LEGAL_QLORA_MEMORY_EFFICIENT",
    "CLASSIFICATION_DEFAULT",
    "NER_DEFAULT",
    "TRAINING_CONFIG_REGISTRY",
    "register_training_config",
    "get_training_config",
    "list_training_configs",
    "resolve_precision",
    "effective_batch_size",
    "to_training_arguments_kwargs",
    "config_to_dict",
    "config_from_dict",
    "save_config",
    "load_config",
    "describe_config",
]