"""
LoRA / QLoRA configuration and model preparation.

Responsibilities:
- Define LoRA configuration.
- Support standard LoRA and QLoRA.
- Prepare Hugging Face models for parameter-efficient fine-tuning.
- Detect suitable target modules.
- Provide trainable-parameter statistics.
- Keep training logic outside this module.

Typical flow:

    Base Model
        │
        ▼
    Quantization (optional)
        │
        ▼
    LoRA Adapter
        │
        ▼
    PEFT Model
        │
        ▼
    Trainer
        │
        ▼
    Adapter Checkpoint
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

try:
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:  # pragma: no cover
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """
    LoRA hyperparameters.

    Defaults are conservative and suitable as a starting point for
    legal-domain instruction/classification experiments.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    bias: str = "none"

    task_type: str = "CAUSAL_LM"

    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )

    modules_to_save: list[str] | None = None

    use_rslora: bool = False
    use_dora: bool = False

    fan_in_fan_out: bool = False

    inference_mode: bool = False

    def __post_init__(self) -> None:

        if self.r <= 0:
            raise ValueError(
                "LoRA rank 'r' must be greater than zero."
            )

        if self.lora_alpha <= 0:
            raise ValueError(
                "lora_alpha must be greater than zero."
            )

        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(
                "lora_dropout must be in [0, 1)."
            )

        if self.bias not in {
            "none",
            "all",
            "lora_only",
        }:
            raise ValueError(
                "bias must be one of: none, all, lora_only."
            )

        if not self.target_modules:
            raise ValueError(
                "At least one target module is required."
            )


@dataclass
class QuantizationConfig:
    """
    QLoRA quantization configuration.

    Requires bitsandbytes when enabled.
    """

    enabled: bool = False

    load_in_4bit: bool = True
    load_in_8bit: bool = False

    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    def __post_init__(self) -> None:

        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError(
                "4-bit and 8-bit quantization cannot both be enabled."
            )

        if self.bnb_4bit_quant_type not in {
            "nf4",
            "fp4",
        }:
            raise ValueError(
                "Unsupported 4-bit quantization type."
            )


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------


def _require_peft() -> None:
    if LoraConfig is None:
        raise ImportError(
            "PEFT is required for LoRA fine-tuning. "
            "Install it with: pip install peft"
        )


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for LoRA fine-tuning."
        )


# ---------------------------------------------------------------------------
# Task mapping
# ---------------------------------------------------------------------------


def resolve_task_type(
    task_type: str,
) -> Any:
    """
    Convert a string task type to PEFT TaskType.
    """

    _require_peft()

    normalized = task_type.strip().upper()

    mapping = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "QUESTION_ANS": TaskType.QUESTION_ANS,
        "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
    }

    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported PEFT task type: {task_type}"
        ) from exc


# ---------------------------------------------------------------------------
# Target-module discovery
# ---------------------------------------------------------------------------


COMMON_TARGET_MODULES: dict[str, list[str]] = {
    "llama": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    "mistral": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    "qwen": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    "gemma": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    "phi": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    "gpt2": [
        "c_attn",
        "c_proj",
    ],
    "bert": [
        "query",
        "key",
        "value",
    ],
}


def infer_target_modules(
    model,
    architecture: str | None = None,
) -> list[str]:
    """
    Infer common attention target modules.

    Explicit target_modules should be preferred for reproducible
    production experiments.
    """

    if architecture:
        key = architecture.lower()

        for family, modules in COMMON_TARGET_MODULES.items():
            if family in key:
                return modules.copy()

    discovered: set[str] = set()

    for name, _module in model.named_modules():

        leaf_name = name.split(".")[-1]

        if leaf_name in {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "query",
            "key",
            "value",
            "c_attn",
            "c_proj",
        }:
            discovered.add(leaf_name)

    if not discovered:
        raise ValueError(
            "Unable to automatically determine LoRA target modules. "
            "Specify target_modules explicitly."
        )

    return sorted(discovered)


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------


def create_lora_config(
    config: LoRAConfig | None = None,
) -> Any:
    """
    Create a PEFT LoraConfig.
    """

    _require_peft()

    config = config or LoRAConfig()

    kwargs: dict[str, Any] = {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": config.bias,
        "task_type": resolve_task_type(
            config.task_type
        ),
        "target_modules": config.target_modules,
        "inference_mode": config.inference_mode,
        "fan_in_fan_out": config.fan_in_fan_out,
    }

    if config.modules_to_save:
        kwargs["modules_to_save"] = config.modules_to_save

    # Newer PEFT versions support RS-LoRA.
    if config.use_rslora:
        kwargs["use_rslora"] = True

    # DoRA is supported by recent PEFT releases.
    if config.use_dora:
        kwargs["use_dora"] = True

    return LoraConfig(**kwargs)


# ---------------------------------------------------------------------------
# Apply LoRA
# ---------------------------------------------------------------------------


def apply_lora(
    model,
    config: LoRAConfig | None = None,
):
    """
    Attach LoRA adapters to a model.
    """

    _require_peft()

    lora_config = create_lora_config(config)

    peft_model = get_peft_model(
        model,
        lora_config,
    )

    return peft_model


# ---------------------------------------------------------------------------
# QLoRA preparation
# ---------------------------------------------------------------------------


def prepare_for_qlora(
    model,
):
    """
    Prepare a quantized model for k-bit LoRA training.
    """

    _require_peft()

    if prepare_model_for_kbit_training is None:
        raise ImportError(
            "Current PEFT installation does not provide "
            "prepare_model_for_kbit_training."
        )

    return prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )


def apply_qlora(
    model,
    config: LoRAConfig | None = None,
):
    """
    Prepare a k-bit model and attach LoRA adapters.
    """

    prepared_model = prepare_for_qlora(model)

    return apply_lora(
        prepared_model,
        config=config,
    )


# ---------------------------------------------------------------------------
# Trainable parameter statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainableParameterStats:
    trainable_parameters: int
    total_parameters: int
    trainable_percentage: float


def get_trainable_parameter_stats(
    model,
) -> TrainableParameterStats:

    trainable = 0
    total = 0

    for parameter in model.parameters():

        parameter_count = parameter.numel()

        total += parameter_count

        if parameter.requires_grad:
            trainable += parameter_count

    percentage = (
        (trainable / total) * 100
        if total
        else 0.0
    )

    return TrainableParameterStats(
        trainable_parameters=trainable,
        total_parameters=total,
        trainable_percentage=percentage,
    )


def print_trainable_parameters(
    model,
) -> TrainableParameterStats:
    """
    Print a concise trainable-parameter summary.
    """

    stats = get_trainable_parameter_stats(model)

    print(
        "Trainable parameters: "
        f"{stats.trainable_parameters:,}"
    )

    print(
        "Total parameters: "
        f"{stats.total_parameters:,}"
    )

    print(
        "Trainable percentage: "
        f"{stats.trainable_percentage:.4f}%"
    )

    return stats


# ---------------------------------------------------------------------------
# Adapter utilities
# ---------------------------------------------------------------------------


def save_lora_adapter(
    model,
    output_dir: str,
) -> None:
    """
    Save only the PEFT adapter weights/configuration.
    """

    if PeftModel is None:
        raise ImportError(
            "PEFT is required."
        )

    if not isinstance(model, PeftModel):
        raise TypeError(
            "Expected a PEFT model."
        )

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )


def merge_lora_adapter(
    model,
):
    """
    Merge LoRA weights into the base model.

    Use this only when the final deployment strategy requires a
    merged model. Keeping adapters separate is often preferable for
    multi-model/multi-tenant experimentation.
    """

    if PeftModel is None:
        raise ImportError(
            "PEFT is required."
        )

    if not isinstance(model, PeftModel):
        raise TypeError(
            "Expected a PEFT model."
        )

    return model.merge_and_unload()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_lora_model(
    model,
) -> None:
    """
    Ensure LoRA actually produced trainable parameters.
    """

    stats = get_trainable_parameter_stats(model)

    if stats.total_parameters == 0:
        raise ValueError(
            "Model contains no parameters."
        )

    if stats.trainable_parameters == 0:
        raise ValueError(
            "No trainable parameters found. "
            "Check LoRA target modules and model configuration."
        )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


__all__ = [
    "LoRAConfig",
    "QuantizationConfig",
    "TrainableParameterStats",
    "COMMON_TARGET_MODULES",
    "create_lora_config",
    "infer_target_modules",
    "apply_lora",
    "prepare_for_qlora",
    "apply_qlora",
    "get_trainable_parameter_stats",
    "print_trainable_parameters",
    "save_lora_adapter",
    "merge_lora_adapter",
    "validate_lora_model",
]