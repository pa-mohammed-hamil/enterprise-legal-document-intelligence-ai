"""
Central Hugging Face model loader.

Path:
    backend/models/huggingface/model_loader.py

Responsibilities:
    - Load Hugging Face tokenizers.
    - Load text-classification models.
    - Load token-classification models.
    - Load sequence-to-sequence / causal language models.
    - Load sentence-transformer style models.
    - Select CPU / CUDA / MPS automatically.
    - Track model revision/version.
    - Support optional quantization configuration.
    - Reuse loaded models through an in-process cache.
    - Provide model metadata.

Non-responsibilities:
    - Business logic.
    - Legal reasoning.
    - Risk decisions.
    - RAG orchestration.
    - Prompt construction.
    - Agent execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Optional

import logging
import os


logger = logging.getLogger(__name__)


class ModelTask(str, Enum):
    """Supported model task categories."""

    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    SEQUENCE_CLASSIFICATION = "sequence-classification"

    CAUSAL_LM = "causal-lm"
    SEQ2SEQ_LM = "seq2seq-lm"

    QUESTION_ANSWERING = "question-answering"

    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"

    EMBEDDING = "embedding"
    RERANKING = "reranking"

    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_TO_TEXT = "image-to-text"

    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"


class DeviceType(str, Enum):
    """Supported execution devices."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for loading one model."""

    model_name: str
    task: ModelTask

    revision: Optional[str] = None
    tokenizer_name: Optional[str] = None

    device: DeviceType | str = DeviceType.AUTO

    trust_remote_code: bool = False
    local_files_only: bool = False

    cache_dir: Optional[str] = None

    torch_dtype: Optional[str] = None

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    use_fast_tokenizer: bool = True

    max_memory: Optional[dict[Any, str]] = None

    model_kwargs: dict[str, Any] = field(
        default_factory=dict
    )

    tokenizer_kwargs: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def effective_tokenizer_name(self) -> str:
        return self.tokenizer_name or self.model_name


@dataclass(slots=True)
class LoadedModel:
    """Loaded model bundle."""

    model: Any
    tokenizer: Any

    model_name: str
    task: ModelTask

    revision: Optional[str]
    device: str

    model_class: str
    tokenizer_class: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def model_id(self) -> str:
        revision = self.revision or "default"

        return (
            f"{self.model_name}"
            f"@{revision}"
            f":{self.task.value}"
        )


class ModelLoader:
    """
    Centralized Hugging Face model loader.

    Example:

        loader = ModelLoader()

        loaded = loader.load(
            ModelConfig(
                model_name="your-org/legal-ner",
                task=ModelTask.TOKEN_CLASSIFICATION,
            )
        )

        model = loaded.model
        tokenizer = loaded.tokenizer
    """

    def __init__(
        self,
        default_device: DeviceType | str = DeviceType.AUTO,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
    ) -> None:
        self.default_device = (
            DeviceType(default_device)
            if isinstance(default_device, str)
            and default_device in {
                item.value for item in DeviceType
            }
            else default_device
        )

        self.cache_dir = (
            cache_dir
            or os.getenv("HF_HOME")
            or os.getenv("TRANSFORMERS_CACHE")
        )

        self.enable_cache = enable_cache

        self._cache: dict[str, LoadedModel] = {}
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        config: ModelConfig,
    ) -> LoadedModel:
        """
        Load a Hugging Face model and tokenizer.
        """

        self._validate_config(config)

        cache_key = self._cache_key(config)

        if self.enable_cache:
            with self._lock:
                cached = self._cache.get(cache_key)

            if cached is not None:
                logger.debug(
                    "Returning cached model: %s",
                    cache_key,
                )
                return cached

        logger.info(
            "Loading model=%s task=%s revision=%s",
            config.model_name,
            config.task.value,
            config.revision,
        )

        tokenizer = self._load_tokenizer(config)

        model = self._load_model(config)

        device = self._resolve_device(
            config.device
        )

        model = self._move_model_to_device(
            model,
            device,
        )

        loaded = LoadedModel(
            model=model,
            tokenizer=tokenizer,
            model_name=config.model_name,
            task=config.task,
            revision=config.revision,
            device=device,
            model_class=model.__class__.__name__,
            tokenizer_class=tokenizer.__class__.__name__,
            metadata={
                "cache_dir": config.cache_dir
                or self.cache_dir,
                "trust_remote_code": (
                    config.trust_remote_code
                ),
                "local_files_only": (
                    config.local_files_only
                ),
                "quantization": self._quantization_info(
                    config
                ),
            },
        )

        if self.enable_cache:
            with self._lock:
                self._cache[cache_key] = loaded

        return loaded

    def unload(
        self,
        config: ModelConfig,
    ) -> bool:
        """
        Remove a model from the in-process cache.

        This does not delete Hugging Face files from disk.
        """

        cache_key = self._cache_key(config)

        with self._lock:
            existed = cache_key in self._cache

            self._cache.pop(
                cache_key,
                None,
            )

        if existed:
            self._release_memory()

        return existed

    def clear_cache(self) -> None:
        """Clear all loaded model references."""

        with self._lock:
            self._cache.clear()

        self._release_memory()

    def get_cached_models(self) -> list[str]:
        """Return currently cached model IDs."""

        with self._lock:
            return [
                loaded.model_id
                for loaded in self._cache.values()
            ]

    def get_model(
        self,
        config: ModelConfig,
    ) -> Any:
        """Return only the model object."""

        return self.load(config).model

    def get_tokenizer(
        self,
        config: ModelConfig,
    ) -> Any:
        """Return only the tokenizer."""

        return self.load(config).tokenizer

    def load_tokenizer(
        self,
        config: ModelConfig,
    ) -> Any:
        """Public tokenizer-only loading API."""

        return self._load_tokenizer(config)

    def resolve_device(
        self,
        device: DeviceType | str | None = None,
    ) -> str:
        """Resolve a requested device."""

        return self._resolve_device(
            device or self.default_device
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _load_tokenizer(
        self,
        config: ModelConfig,
    ) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required to load "
                "Hugging Face tokenizers."
            ) from exc

        kwargs = dict(
            config.tokenizer_kwargs
        )

        kwargs.setdefault(
            "use_fast",
            config.use_fast_tokenizer,
        )

        kwargs.setdefault(
            "trust_remote_code",
            config.trust_remote_code,
        )

        kwargs.setdefault(
            "local_files_only",
            config.local_files_only,
        )

        if config.revision is not None:
            kwargs["revision"] = config.revision

        cache_dir = (
            config.cache_dir
            or self.cache_dir
        )

        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        return AutoTokenizer.from_pretrained(
            config.effective_tokenizer_name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(
        self,
        config: ModelConfig,
    ) -> Any:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageClassification,
                AutoModelForQuestionAnswering,
                AutoModelForSeq2SeqLM,
                AutoModelForSequenceClassification,
                AutoModelForTokenClassification,
                AutoModelForVision2Seq,
            )
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required "
                "to load Hugging Face models."
            ) from exc

        model_kwargs = dict(
            config.model_kwargs
        )

        model_kwargs.setdefault(
            "trust_remote_code",
            config.trust_remote_code,
        )

        model_kwargs.setdefault(
            "local_files_only",
            config.local_files_only,
        )

        if config.revision is not None:
            model_kwargs["revision"] = config.revision

        cache_dir = (
            config.cache_dir
            or self.cache_dir
        )

        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir

        dtype = self._resolve_dtype(
            config.torch_dtype,
            torch,
        )

        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        quantization = self._build_quantization_config(
            config,
        )

        if quantization is not None:
            model_kwargs[
                "quantization_config"
            ] = quantization

        if config.max_memory:
            model_kwargs["max_memory"] = (
                config.max_memory
            )

        task = config.task

        if task == ModelTask.CAUSAL_LM:
            model_class = AutoModelForCausalLM

        elif task == ModelTask.SEQ2SEQ_LM:
            model_class = AutoModelForSeq2SeqLM

        elif task in {
            ModelTask.TEXT_CLASSIFICATION,
            ModelTask.SEQUENCE_CLASSIFICATION,
            ModelTask.ZERO_SHOT_CLASSIFICATION,
        }:
            model_class = (
                AutoModelForSequenceClassification
            )

        elif task == ModelTask.TOKEN_CLASSIFICATION:
            model_class = (
                AutoModelForTokenClassification
            )

        elif task == ModelTask.QUESTION_ANSWERING:
            model_class = (
                AutoModelForQuestionAnswering
            )

        elif task == ModelTask.IMAGE_CLASSIFICATION:
            model_class = (
                AutoModelForImageClassification
            )

        elif task == ModelTask.IMAGE_TO_TEXT:
            model_class = AutoModelForVision2Seq

        elif task in {
            ModelTask.EMBEDDING,
            ModelTask.RERANKING,
        }:
            return self._load_embedding_or_reranker(
                config
            )

        else:
            raise ValueError(
                f"Unsupported model task: {task}"
            )

        model = model_class.from_pretrained(
            config.model_name,
            **model_kwargs,
        )

        # Avoid calling .to() on quantized models.
        if not (
            config.load_in_4bit
            or config.load_in_8bit
        ):
            model.eval()

        return model

    # ------------------------------------------------------------------
    # Sentence Transformers
    # ------------------------------------------------------------------

    def _load_embedding_or_reranker(
        self,
        config: ModelConfig,
    ) -> Any:
        """
        Load embedding/reranker models.

        SentenceTransformer is deliberately isolated from
        transformers AutoModel loading.
        """

        try:
            from sentence_transformers import (
                SentenceTransformer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required "
                "for embedding/reranking models."
            ) from exc

        device = self._resolve_device(
            config.device
        )

        kwargs = dict(
            config.model_kwargs
        )

        if config.cache_dir or self.cache_dir:
            kwargs["cache_folder"] = (
                config.cache_dir
                or self.cache_dir
            )

        model = SentenceTransformer(
            config.model_name,
            device=device,
            **kwargs,
        )

        return model

    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------

    def _resolve_device(
        self,
        device: DeviceType | str,
    ) -> str:
        requested = (
            device.value
            if isinstance(device, DeviceType)
            else str(device).lower()
        )

        if requested == DeviceType.AUTO.value:
            return self._auto_device()

        if requested == DeviceType.CUDA.value:
            if self._cuda_available():
                return "cuda"

            logger.warning(
                "CUDA requested but unavailable; "
                "falling back to CPU."
            )

            return "cpu"

        if requested == DeviceType.MPS.value:
            if self._mps_available():
                return "mps"

            logger.warning(
                "MPS requested but unavailable; "
                "falling back to CPU."
            )

            return "cpu"

        if requested == DeviceType.CPU.value:
            return "cpu"

        # Permit explicit CUDA device IDs such as cuda:1.
        if requested.startswith("cuda:"):
            if self._cuda_available():
                return requested

            return "cpu"

        raise ValueError(
            f"Unsupported device: {device}"
        )

    def _auto_device(self) -> str:
        if self._cuda_available():
            return "cuda"

        if self._mps_available():
            return "mps"

        return "cpu"

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(
                torch.cuda.is_available()
            )
        except ImportError:
            return False

    @staticmethod
    def _mps_available() -> bool:
        try:
            import torch

            return bool(
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        except ImportError:
            return False

    @staticmethod
    def _move_model_to_device(
        model: Any,
        device: str,
    ) -> Any:
        """
        Move ordinary PyTorch models.

        SentenceTransformer models generally already receive
        their device during construction.
        """

        if device == "cpu":
            try:
                return model.to("cpu")
            except AttributeError:
                return model

        # Quantized models may manage their own device map.
        if hasattr(model, "hf_device_map"):
            return model

        try:
            return model.to(device)
        except AttributeError:
            return model

    # ------------------------------------------------------------------
    # Dtype / quantization
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_dtype(
        dtype: Optional[str],
        torch: Any,
    ) -> Any:
        if dtype is None:
            return None

        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }

        normalized = dtype.lower()

        if normalized not in mapping:
            raise ValueError(
                f"Unsupported torch dtype: {dtype}"
            )

        return mapping[normalized]

    def _build_quantization_config(
        self,
        config: ModelConfig,
    ) -> Any:
        if not (
            config.load_in_4bit
            or config.load_in_8bit
        ):
            return None

        try:
            from transformers import (
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "bitsandbytes/transformers quantization "
                "support is required."
            ) from exc

        kwargs: dict[str, Any] = {}

        if config.load_in_4bit:
            kwargs.update(
                {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": self._get_bf16_or_fp16(),
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                }
            )

        if config.load_in_8bit:
            kwargs["load_in_8bit"] = True

        return BitsAndBytesConfig(
            **kwargs
        )

    @staticmethod
    def _get_bf16_or_fp16() -> Any:
        try:
            import torch

            if (
                torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
            ):
                return torch.bfloat16

            return torch.float16

        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Validation / cache
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_config(
        config: ModelConfig,
    ) -> None:
        if not isinstance(
            config,
            ModelConfig,
        ):
            raise TypeError(
                "config must be a ModelConfig instance."
            )

        if not config.model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if (
            config.load_in_4bit
            and config.load_in_8bit
        ):
            raise ValueError(
                "load_in_4bit and load_in_8bit "
                "cannot both be enabled."
            )

    @staticmethod
    def _cache_key(
        config: ModelConfig,
    ) -> str:
        revision = config.revision or "default"
        device = str(config.device)

        quantization = (
            "4bit"
            if config.load_in_4bit
            else "8bit"
            if config.load_in_8bit
            else "none"
        )

        return (
            f"{config.model_name}"
            f"|{revision}"
            f"|{config.task.value}"
            f"|{device}"
            f"|{quantization}"
        )

    @staticmethod
    def _quantization_info(
        config: ModelConfig,
    ) -> str:
        if config.load_in_4bit:
            return "4bit"

        if config.load_in_8bit:
            return "8bit"

        return "none"

    @staticmethod
    def _release_memory() -> None:
        """Best-effort GPU memory cleanup."""

        try:
            import gc

            gc.collect()

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except ImportError:
            pass


# ----------------------------------------------------------------------
# Convenience factory
# ----------------------------------------------------------------------

def create_model_loader(
    *,
    device: DeviceType | str = DeviceType.AUTO,
    cache_dir: Optional[str] = None,
    enable_cache: bool = True,
) -> ModelLoader:
    """
    Create the application's shared model loader.

    Keeping this factory small makes it easy to replace with
    dependency injection later.
    """

    return ModelLoader(
        default_device=device,
        cache_dir=cache_dir,
        enable_cache=enable_cache,
    )


__all__ = [
    "ModelTask",
    "DeviceType",
    "ModelConfig",
    "LoadedModel",
    "ModelLoader",
    "create_model_loader",
]