"""
backend/models/fine_tuning/peft_config.py

Central PEFT configuration registry for legal-domain fine-tuning.

Responsibilities
----------------
- Define PEFT methods.
- Define supported fine-tuning tasks.
- Define reusable LoRA / QLoRA presets.
- Validate PEFT configurations.
- Convert configurations into LoRA/quantization arguments.
- Provide a central experiment configuration registry.

This module does NOT:
- Load models.
- Tokenize datasets.
- Train models.
- Apply adapters.

Those responsibilities belong to:
    lora.py
    dataset_builder.py
    trainer.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


# ============================================================================
# ENUMS
# ============================================================================


class PEFTMethod(str, Enum):
    """Supported parameter-efficient fine-tuning methods."""

    LORA = "lora"
    QLORA = "qlora"


class FineTuningTask(str, Enum):
    """Supported fine-tuning tasks in the legal AI platform."""

    LEGAL_QA = "legal_qa"
    DOCUMENT_CLASSIFICATION = "document_classification"
    CLAUSE_CLASSIFICATION = "clause_classification"
    LEGAL_NER = "legal_ner"
    RISK_CLASSIFICATION = "risk_classification"
    CONTRACT_COMPARISON = "contract_comparison"
    INSTRUCTION_TUNING = "instruction_tuning"


class ModelFamily(str, Enum):
    """Common model families supported by the project."""

    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    GEMMA = "gemma"
    PHI = "phi"
    GPT2 = "gpt2"
    BERT = "bert"
    GENERIC = "generic"


# ============================================================================
# TARGET MODULE PRESETS
# ============================================================================


TARGET_MODULE_PRESETS: dict[
    ModelFamily,
    tuple[str, ...],
] = {
    ModelFamily.LLAMA: (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ),
    ModelFamily.MISTRAL: (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ),
    ModelFamily.QWEN: (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ),
    ModelFamily.GEMMA: (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ),
    ModelFamily.PHI: (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ),
    ModelFamily.GPT2: (
        "c_attn",
        "c_proj",
    ),
    ModelFamily.BERT: (
        "query",
        "key",
        "value",
    ),
}


# ============================================================================
# PEFT CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class PEFTConfig:
    """
    Complete configuration for one PEFT experiment.

    The object is intentionally framework-agnostic enough to be stored
    in experiment metadata and later translated into PEFT arguments.
    """

    name: str

    method: PEFTMethod = PEFTMethod.LORA

    task: FineTuningTask = FineTuningTask.LEGAL_QA

    model_family: ModelFamily = ModelFamily.LLAMA

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    r: int = 16

    lora_alpha: int = 32

    lora_dropout: float = 0.05

    bias: str = "none"

    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )

    modules_to_save: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # Advanced LoRA
    # ------------------------------------------------------------------

    use_rslora: bool = False

    use_dora: bool = False

    fan_in_fan_out: bool = False

    # ------------------------------------------------------------------
    # QLoRA
    # ------------------------------------------------------------------

    load_in_4bit: bool = False

    load_in_8bit: bool = False

    bnb_4bit_quant_type: str = "nf4"

    bnb_4bit_compute_dtype: str = "bfloat16"

    bnb_4bit_use_double_quant: bool = True

    # ------------------------------------------------------------------
    # Training behavior
    # ------------------------------------------------------------------

    gradient_checkpointing: bool = True

    inference_mode: bool = False

    # ------------------------------------------------------------------
    # Experiment metadata
    # ------------------------------------------------------------------

    description: str = ""

    tags: tuple[str, ...] = field(
        default_factory=tuple
    )

    version: str = "1.0.0"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:

        if not self.name.strip():
            raise ValueError(
                "PEFT configuration name cannot be empty."
            )

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
                "lora_dropout must be in the range [0, 1)."
            )

        if self.bias not in {
            "none",
            "all",
            "lora_only",
        }:
            raise ValueError(
                "bias must be one of: "
                "'none', 'all', 'lora_only'."
            )

        if not self.target_modules:
            raise ValueError(
                "At least one target module is required."
            )

        if (
            self.load_in_4bit
            and self.load_in_8bit
        ):
            raise ValueError(
                "4-bit and 8-bit quantization "
                "cannot both be enabled."
            )

        if self.bnb_4bit_quant_type not in {
            "nf4",
            "fp4",
        }:
            raise ValueError(
                "bnb_4bit_quant_type must be "
                "'nf4' or 'fp4'."
            )

        if (
            self.method == PEFTMethod.QLORA
            and not (
                self.load_in_4bit
                or self.load_in_8bit
            )
        ):
            raise ValueError(
                "QLoRA requires either 4-bit "
                "or 8-bit quantization."
            )

        if (
            self.method == PEFTMethod.LORA
            and (
                self.load_in_4bit
                or self.load_in_8bit
            )
        ):
            raise ValueError(
                "Standard LoRA should not enable "
                "QLoRA quantization flags."
            )


# ============================================================================
# FACTORY
# ============================================================================


def create_peft_config(
    *,
    name: str,
    task: FineTuningTask,
    model_family: ModelFamily = ModelFamily.LLAMA,
    method: PEFTMethod = PEFTMethod.LORA,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: tuple[str, ...] | None = None,
    modules_to_save: tuple[str, ...] = (),
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    use_rslora: bool = False,
    use_dora: bool = False,
    gradient_checkpointing: bool = True,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> PEFTConfig:
    """
    Create a validated PEFT configuration.

    If target_modules are omitted, the model-family defaults are used.
    """

    if target_modules is None:

        target_modules = TARGET_MODULE_PRESETS.get(
            model_family,
            (),
        )

    if not target_modules:
        raise ValueError(
            "target_modules must be supplied for "
            f"model family '{model_family.value}'."
        )

    return PEFTConfig(
        name=name,
        method=method,
        task=task,
        model_family=model_family,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        use_rslora=use_rslora,
        use_dora=use_dora,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        gradient_checkpointing=gradient_checkpointing,
        description=description,
        tags=tags,
    )


# ============================================================================
# PRODUCTION PRESETS
# ============================================================================


LEGAL_QA_LORA = create_peft_config(
    name="legal-qa-lora",
    task=FineTuningTask.LEGAL_QA,
    method=PEFTMethod.LORA,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    description=(
        "LoRA configuration for domain-specific legal "
        "question answering and grounded instruction tuning."
    ),
    tags=(
        "legal",
        "qa",
        "rag",
        "instruction",
    ),
)


LEGAL_QA_QLORA = create_peft_config(
    name="legal-qa-qlora",
    task=FineTuningTask.LEGAL_QA,
    method=PEFTMethod.QLORA,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    load_in_4bit=True,
    description=(
        "Memory-efficient QLoRA configuration for legal "
        "domain adaptation."
    ),
    tags=(
        "legal",
        "qa",
        "qlora",
        "4bit",
    ),
)


LEGAL_INSTRUCTION_QLORA = create_peft_config(
    name="legal-instruction-qlora",
    task=FineTuningTask.INSTRUCTION_TUNING,
    method=PEFTMethod.QLORA,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    load_in_4bit=True,
    description=(
        "Higher-capacity QLoRA configuration for "
        "legal-domain instruction tuning."
    ),
    tags=(
        "legal",
        "instruction",
        "qlora",
    ),
)


CONTRACT_COMPARISON_LORA = create_peft_config(
    name="contract-comparison-lora",
    task=FineTuningTask.CONTRACT_COMPARISON,
    method=PEFTMethod.LORA,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    description=(
        "LoRA configuration for structured contract "
        "comparison and change analysis."
    ),
    tags=(
        "contracts",
        "comparison",
        "analysis",
    ),
)


# ============================================================================
# CLASSIFICATION PRESETS
# ============================================================================


DOCUMENT_CLASSIFIER_LORA = create_peft_config(
    name="document-classifier-lora",
    task=FineTuningTask.DOCUMENT_CLASSIFICATION,
    model_family=ModelFamily.BERT,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=(
        "query",
        "key",
        "value",
    ),
    description=(
        "LoRA configuration for legal document "
        "type classification."
    ),
    tags=(
        "document",
        "classification",
    ),
)


CLAUSE_CLASSIFIER_LORA = create_peft_config(
    name="clause-classifier-lora",
    task=FineTuningTask.CLAUSE_CLASSIFICATION,
    model_family=ModelFamily.BERT,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=(
        "query",
        "key",
        "value",
    ),
    description=(
        "LoRA configuration for legal clause "
        "classification."
    ),
    tags=(
        "clause",
        "classification",
    ),
)


RISK_CLASSIFIER_LORA = create_peft_config(
    name="risk-classifier-lora",
    task=FineTuningTask.RISK_CLASSIFICATION,
    model_family=ModelFamily.BERT,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=(
        "query",
        "key",
        "value",
    ),
    description=(
        "LoRA configuration for contract risk "
        "classification."
    ),
    tags=(
        "risk",
        "classification",
        "legal",
    ),
)


LEGAL_NER_LORA = create_peft_config(
    name="legal-ner-lora",
    task=FineTuningTask.LEGAL_NER,
    model_family=ModelFamily.BERT,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=(
        "query",
        "key",
        "value",
    ),
    description=(
        "LoRA configuration for legal named "
        "entity recognition."
    ),
    tags=(
        "ner",
        "legal-entities",
    ),
)


# ============================================================================
# REGISTRY
# ============================================================================


PEFT_CONFIG_REGISTRY: dict[str, PEFTConfig] = {
    config.name: config
    for config in (
        LEGAL_QA_LORA,
        LEGAL_QA_QLORA,
        LEGAL_INSTRUCTION_QLORA,
        CONTRACT_COMPARISON_LORA,
        DOCUMENT_CLASSIFIER_LORA,
        CLAUSE_CLASSIFIER_LORA,
        RISK_CLASSIFIER_LORA,
        LEGAL_NER_LORA,
    )
}


# ============================================================================
# REGISTRY OPERATIONS
# ============================================================================


def register_peft_config(
    config: PEFTConfig,
    *,
    overwrite: bool = False,
) -> None:
    """
    Register a custom experiment configuration.
    """

    validate_config(config)

    if (
        config.name in PEFT_CONFIG_REGISTRY
        and not overwrite
    ):
        raise ValueError(
            f"PEFT configuration '{config.name}' "
            "already exists."
        )

    PEFT_CONFIG_REGISTRY[config.name] = config


def get_peft_config(
    name: str,
) -> PEFTConfig:
    """
    Retrieve a configuration by name.
    """

    try:
        return PEFT_CONFIG_REGISTRY[name]
    except KeyError as exc:

        available = ", ".join(
            sorted(
                PEFT_CONFIG_REGISTRY.keys()
            )
        )

        raise KeyError(
            f"Unknown PEFT configuration '{name}'. "
            f"Available configurations: {available}"
        ) from exc


def list_peft_configs() -> list[str]:
    """Return all registered configuration names."""

    return sorted(
        PEFT_CONFIG_REGISTRY.keys()
    )


def get_configs_for_task(
    task: FineTuningTask | str,
) -> list[PEFTConfig]:
    """
    Return all configurations registered for a task.
    """

    if isinstance(task, str):
        task = FineTuningTask(task)

    return [
        config
        for config in PEFT_CONFIG_REGISTRY.values()
        if config.task == task
    ]


# ============================================================================
# MODEL TARGET HELPERS
# ============================================================================


def get_target_modules(
    model_family: ModelFamily | str,
) -> tuple[str, ...]:
    """
    Return recommended LoRA target modules for a model family.
    """

    if isinstance(model_family, str):
        model_family = ModelFamily(model_family)

    try:
        return TARGET_MODULE_PRESETS[
            model_family
        ]
    except KeyError as exc:
        raise ValueError(
            f"No target-module preset exists for "
            f"model family '{model_family.value}'. "
            "Specify target_modules manually."
        ) from exc


# ============================================================================
# CONVERSION
# ============================================================================


def to_lora_kwargs(
    config: PEFTConfig,
) -> dict[str, Any]:
    """
    Convert the project configuration into arguments
    consumed by the LoRA implementation.
    """

    validate_config(config)

    return {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": config.bias,
        "target_modules": list(
            config.target_modules
        ),
        "modules_to_save": (
            list(config.modules_to_save)
            if config.modules_to_save
            else None
        ),
        "use_rslora": config.use_rslora,
        "use_dora": config.use_dora,
        "fan_in_fan_out": config.fan_in_fan_out,
        "inference_mode": config.inference_mode,
        "task_type": config.task.value,
    }


def to_quantization_kwargs(
    config: PEFTConfig,
) -> dict[str, Any]:
    """
    Convert QLoRA configuration into model-loading kwargs.
    """

    validate_config(config)

    if config.method != PEFTMethod.QLORA:
        return {}

    kwargs: dict[str, Any] = {
        "load_in_4bit": config.load_in_4bit,
        "load_in_8bit": config.load_in_8bit,
    }

    if config.load_in_4bit:

        kwargs.update(
            {
                "bnb_4bit_quant_type": (
                    config.bnb_4bit_quant_type
                ),
                "bnb_4bit_compute_dtype": (
                    config.bnb_4bit_compute_dtype
                ),
                "bnb_4bit_use_double_quant": (
                    config.bnb_4bit_use_double_quant
                ),
            }
        )

    return kwargs


# ============================================================================
# SERIALIZATION
# ============================================================================


def config_to_dict(
    config: PEFTConfig,
) -> dict[str, Any]:
    """
    Serialize a configuration for experiment tracking.
    """

    data = asdict(config)

    data["method"] = config.method.value
    data["task"] = config.task.value
    data["model_family"] = (
        config.model_family.value
    )

    data["target_modules"] = list(
        config.target_modules
    )

    data["modules_to_save"] = list(
        config.modules_to_save
    )

    data["tags"] = list(
        config.tags
    )

    return data


def config_from_dict(
    data: dict[str, Any],
) -> PEFTConfig:
    """
    Reconstruct a PEFTConfig from serialized metadata.
    """

    payload = dict(data)

    payload["method"] = PEFTMethod(
        payload["method"]
    )

    payload["task"] = FineTuningTask(
        payload["task"]
    )

    payload["model_family"] = ModelFamily(
        payload["model_family"]
    )

    payload["target_modules"] = tuple(
        payload.get(
            "target_modules",
            (),
        )
    )

    payload["modules_to_save"] = tuple(
        payload.get(
            "modules_to_save",
            (),
        )
    )

    payload["tags"] = tuple(
        payload.get(
            "tags",
            (),
        )
    )

    return PEFTConfig(
        **payload
    )


# ============================================================================
# VALIDATION
# ============================================================================


def validate_config(
    config: PEFTConfig,
) -> None:
    """
    Validate configuration against PEFT invariants.
    """

    if not isinstance(
        config,
        PEFTConfig,
    ):
        raise TypeError(
            "Expected a PEFTConfig instance."
        )

    if (
        config.method == PEFTMethod.QLORA
        and not (
            config.load_in_4bit
            or config.load_in_8bit
        )
    ):
        raise ValueError(
            "QLoRA requires 4-bit or 8-bit quantization."
        )

    if (
        config.method == PEFTMethod.LORA
        and (
            config.load_in_4bit
            or config.load_in_8bit
        )
    ):
        raise ValueError(
            "Standard LoRA cannot enable QLoRA "
            "quantization settings."
        )


# ============================================================================
# EXPERIMENT SUMMARY
# ============================================================================


def describe_config(
    config: PEFTConfig,
) -> dict[str, Any]:
    """
    Return a compact human/experiment-readable summary.
    """

    return {
        "name": config.name,
        "method": config.method.value,
        "task": config.task.value,
        "model_family": config.model_family.value,
        "rank": config.r,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "target_modules": list(
            config.target_modules
        ),
        "quantized": (
            config.load_in_4bit
            or config.load_in_8bit
        ),
        "gradient_checkpointing": (
            config.gradient_checkpointing
        ),
        "tags": list(config.tags),
    }


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "PEFTMethod",
    "FineTuningTask",
    "ModelFamily",
    "PEFTConfig",
    "TARGET_MODULE_PRESETS",
    "LEGAL_QA_LORA",
    "LEGAL_QA_QLORA",
    "LEGAL_INSTRUCTION_QLORA",
    "CONTRACT_COMPARISON_LORA",
    "DOCUMENT_CLASSIFIER_LORA",
    "CLAUSE_CLASSIFIER_LORA",
    "RISK_CLASSIFIER_LORA",
    "LEGAL_NER_LORA",
    "PEFT_CONFIG_REGISTRY",
    "create_peft_config",
    "register_peft_config",
    "get_peft_config",
    "list_peft_configs",
    "get_configs_for_task",
    "get_target_modules",
    "to_lora_kwargs",
    "to_quantization_kwargs",
    "config_to_dict",
    "config_from_dict",
    "validate_config",
    "describe_config",
]