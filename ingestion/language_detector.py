"""
Document language detection.

Path:
    backend/ingestion/language_detector.py

Responsibilities:
    - Detect document language.
    - Estimate detection confidence.
    - Detect dominant writing script.
    - Detect mixed-language documents.
    - Provide deterministic fallback behavior.
    - Support injectable production language detectors.

Non-responsibilities:
    - Translation.
    - OCR.
    - Text normalization.
    - Legal classification.
    - Legal interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
import logging
import re
import unicodedata


logger = logging.getLogger(__name__)


# ======================================================================
# Enums
# ======================================================================


class LanguageCode(str, Enum):
    UNKNOWN = "unknown"

    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    POLISH = "pl"
    RUSSIAN = "ru"
    UKRAINIAN = "uk"

    HINDI = "hi"
    MALAYALAM = "ml"
    TAMIL = "ta"
    TELUGU = "te"
    KANNADA = "kn"
    BENGALI = "bn"
    MARATHI = "mr"
    GUJARATI = "gu"
    PUNJABI = "pa"

    ARABIC = "ar"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"


class ScriptCode(str, Enum):
    UNKNOWN = "unknown"

    LATIN = "latin"
    DEVANAGARI = "devanagari"
    MALAYALAM = "malayalam"
    TAMIL = "tamil"
    TELUGU = "telugu"
    KANNADA = "kannada"
    BENGALI = "bengali"
    GURMUKHI = "gurmukhi"
    GUJARATI = "gujarati"
    ARABIC = "arabic"
    CYRILLIC = "cyrillic"
    HAN = "han"
    HIRAGANA = "hiragana"
    KATAKANA = "katakana"
    HANGUL = "hangul"


class DetectionMethod(str, Enum):
    INJECTED = "injected"
    SCRIPT = "script"
    HEURISTIC = "heuristic"
    FALLBACK = "fallback"


# ======================================================================
# Data structures
# ======================================================================


@dataclass(slots=True)
class LanguageScore:
    language: LanguageCode

    score: float

    evidence: list[str] = field(
        default_factory=list
    )


@dataclass(slots=True)
class LanguageDetectionResult:
    language: LanguageCode

    confidence: float

    script: ScriptCode

    method: DetectionMethod

    is_mixed: bool = False

    languages: list[LanguageCode] = field(
        default_factory=list
    )

    language_scores: list[LanguageScore] = field(
        default_factory=list
    )

    character_count: int = 0

    analyzed_character_count: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )


@dataclass(slots=True)
class LanguageDetectorConfig:
    min_text_characters: int = 20

    max_analysis_characters: int = 50_000

    confidence_threshold: float = 0.60

    mixed_language_threshold: float = 0.20

    max_languages: int = 5

    ignore_numbers: bool = True

    ignore_punctuation: bool = True

    enable_script_detection: bool = True

    enable_heuristic_detection: bool = True

    # A very small amount of text is unreliable.
    short_text_confidence_cap: float = 0.55


# ======================================================================
# Language Detector
# ======================================================================


class LanguageDetector:
    """
    Production-friendly language detection abstraction.

    Preferred deployment:

        LanguageDetector(
            detector=my_fasttext_or_lingua_detector
        )

    The injected detector may return:

        "en"

    or:

        {
            "language": "en",
            "confidence": 0.97,
        }

    or:

        [
            ("en", 0.97),
            ("de", 0.02),
        ]
    """

    # --------------------------------------------------------------
    # Common stop words / legal vocabulary.
    # --------------------------------------------------------------

    LANGUAGE_MARKERS: dict[
        LanguageCode,
        tuple[str, ...],
    ] = {
        LanguageCode.ENGLISH: (
            "the",
            "and",
            "or",
            "of",
            "to",
            "in",
            "for",
            "with",
            "agreement",
            "party",
            "parties",
            "shall",
            "may",
            "must",
            "contract",
            "section",
            "clause",
            "notice",
            "law",
            "services",
            "payment",
            "confidential",
            "termination",
        ),
        LanguageCode.GERMAN: (
            "der",
            "die",
            "das",
            "und",
            "oder",
            "von",
            "zu",
            "für",
            "mit",
            "vertrag",
            "partei",
            "parteien",
            "abschnitt",
            "kündigung",
            "zahlung",
            "gesetz",
            "dienstleistungen",
        ),
        LanguageCode.FRENCH: (
            "le",
            "la",
            "les",
            "des",
            "et",
            "ou",
            "de",
            "pour",
            "avec",
            "contrat",
            "partie",
            "parties",
            "article",
            "résiliation",
            "paiement",
            "loi",
        ),
        LanguageCode.SPANISH: (
            "el",
            "la",
            "los",
            "las",
            "y",
            "o",
            "de",
            "para",
            "con",
            "contrato",
            "parte",
            "partes",
            "sección",
            "terminación",
            "pago",
            "ley",
        ),
        LanguageCode.ITALIAN: (
            "il",
            "lo",
            "la",
            "gli",
            "le",
            "e",
            "o",
            "di",
            "per",
            "con",
            "contratto",
            "parte",
            "parti",
            "sezione",
            "pagamento",
            "legge",
        ),
        LanguageCode.PORTUGUESE: (
            "o",
            "a",
            "os",
            "as",
            "e",
            "ou",
            "de",
            "para",
            "com",
            "contrato",
            "parte",
            "partes",
            "seção",
            "pagamento",
            "lei",
        ),
        LanguageCode.DUTCH: (
            "de",
            "het",
            "een",
            "en",
            "of",
            "van",
            "voor",
            "met",
            "overeenkomst",
            "partij",
            "partijen",
            "betaling",
            "wet",
        ),
        LanguageCode.HINDI: (
            "और",
            "या",
            "का",
            "के",
            "की",
            "में",
            "से",
            "लिए",
            "यह",
            "है",
            "अनुबंध",
            "पक्ष",
            "धारा",
            "भुगतान",
        ),
        LanguageCode.MALAYALAM: (
            "ആണ്",
            "ഒരു",
            "ഈ",
            "അത്",
            "എന്ന",
            "എന്നാൽ",
            "കൂടാതെ",
            "കരാർ",
            "കക്ഷി",
            "വകുപ്പ്",
            "പണം",
            "നിയമം",
        ),
        LanguageCode.TAMIL: (
            "மற்றும்",
            "அல்லது",
            "ஒரு",
            "இந்த",
            "அது",
            "உள்ள",
            "ஒப்பந்தம்",
            "தரப்பு",
            "பிரிவு",
            "பணம்",
            "சட்டம்",
        ),
        LanguageCode.TELUGU: (
            "మరియు",
            "లేదా",
            "ఒక",
            "ఈ",
            "అది",
            "లో",
            "కాంట్రాక్ట్",
            "పక్షం",
            "చెల్లింపు",
            "చట్టం",
        ),
    }

    def __init__(
        self,
        *,
        config: Optional[
            LanguageDetectorConfig
        ] = None,
        detector: Optional[
            Callable[[str], Any]
        ] = None,
    ) -> None:
        self.config = (
            config
            or LanguageDetectorConfig()
        )

        self.detector = detector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        text: str,
    ) -> LanguageDetectionResult:
        """
        Detect dominant document language.
        """

        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "text must be a string"
            )

        original_length = len(text)

        if not text.strip():
            return LanguageDetectionResult(
                language=LanguageCode.UNKNOWN,
                confidence=0.0,
                script=ScriptCode.UNKNOWN,
                method=DetectionMethod.FALLBACK,
                character_count=0,
                analyzed_character_count=0,
                warnings=[
                    "Document contains no text."
                ],
            )

        analysis_text = self._prepare_text(
            text
        )

        if len(analysis_text) < (
            self.config.min_text_characters
        ):
            result = self._detect_short_text(
                analysis_text
            )

            result.character_count = (
                original_length
            )

            result.analyzed_character_count = (
                len(analysis_text)
            )

            return result

        # --------------------------------------------------------------
        # Script detection first.
        # --------------------------------------------------------------

        script = (
            self.detect_script(
                analysis_text
            )
            if self.config.enable_script_detection
            else ScriptCode.UNKNOWN
        )

        # --------------------------------------------------------------
        # Injected production detector.
        # --------------------------------------------------------------

        if self.detector:
            try:
                result = (
                    self._detect_with_injected_detector(
                        analysis_text
                    )
                )

                result.script = script
                result.character_count = (
                    original_length
                )
                result.analyzed_character_count = (
                    len(analysis_text)
                )
                result.method = (
                    DetectionMethod.INJECTED
                )

                result.is_mixed = (
                    self._is_mixed_language(
                        result
                    )
                )

                return result

            except Exception as exc:
                logger.warning(
                    "Injected language detector failed: %s",
                    exc,
                )

        # --------------------------------------------------------------
        # Script-based detection.
        # --------------------------------------------------------------

        script_language = (
            self._language_from_script(
                script
            )
        )

        # Some scripts map nearly one-to-one
        # to languages.
        if script_language:
            confidence = (
                self._script_confidence(
                    analysis_text,
                    script,
                )
            )

            if confidence >= (
                self.config.confidence_threshold
            ):
                return LanguageDetectionResult(
                    language=script_language,
                    confidence=confidence,
                    script=script,
                    method=DetectionMethod.SCRIPT,
                    is_mixed=False,
                    languages=[
                        script_language
                    ],
                    character_count=original_length,
                    analyzed_character_count=(
                        len(analysis_text)
                    ),
                )

        # --------------------------------------------------------------
        # Heuristic fallback.
        # --------------------------------------------------------------

        if self.config.enable_heuristic_detection:
            result = (
                self._heuristic_detect(
                    analysis_text,
                    script,
                )
            )

            result.character_count = (
                original_length
            )

            result.analyzed_character_count = (
                len(analysis_text)
            )

            return result

        return LanguageDetectionResult(
            language=LanguageCode.UNKNOWN,
            confidence=0.0,
            script=script,
            method=DetectionMethod.FALLBACK,
            character_count=original_length,
            analyzed_character_count=(
                len(analysis_text)
            ),
            warnings=[
                "No language detector is configured."
            ],
        )

    # ------------------------------------------------------------------
    # Script detection
    # ------------------------------------------------------------------

    def detect_script(
        self,
        text: str,
    ) -> ScriptCode:
        """
        Detect dominant writing script.
        """

        counts: dict[
            ScriptCode,
            int,
        ] = {}

        for character in text:
            script = self._character_script(
                character
            )

            if script == ScriptCode.UNKNOWN:
                continue

            counts[script] = (
                counts.get(
                    script,
                    0,
                )
                + 1
            )

        if not counts:
            return ScriptCode.UNKNOWN

        return max(
            counts,
            key=counts.get,
        )

    def script_distribution(
        self,
        text: str,
    ) -> dict[
        ScriptCode,
        float,
    ]:
        """
        Return script proportions.
        """

        counts: dict[
            ScriptCode,
            int,
        ] = {}

        total = 0

        for character in text:
            script = self._character_script(
                character
            )

            if script == ScriptCode.UNKNOWN:
                continue

            counts[script] = (
                counts.get(
                    script,
                    0,
                )
                + 1
            )

            total += 1

        if total == 0:
            return {}

        return {
            script: round(
                count / total,
                4,
            )
            for script, count in counts.items()
        }

    # ------------------------------------------------------------------
    # Injected detector
    # ------------------------------------------------------------------

    def _detect_with_injected_detector(
        self,
        text: str,
    ) -> LanguageDetectionResult:
        raw = self.detector(
            text
        )

        if isinstance(
            raw,
            str,
        ):
            language = (
                self._normalize_language_code(
                    raw
                )
            )

            return LanguageDetectionResult(
                language=language,
                confidence=0.80,
                script=self.detect_script(
                    text
                ),
                method=DetectionMethod.INJECTED,
                languages=[
                    language
                ],
            )

        if isinstance(
            raw,
            dict,
        ):
            language = (
                self._normalize_language_code(
                    raw.get(
                        "language",
                        raw.get(
                            "lang",
                            "unknown",
                        ),
                    )
                )
            )

            confidence = float(
                raw.get(
                    "confidence",
                    raw.get(
                        "probability",
                        0.0,
                    ),
                )
            )

            languages = [
                self._normalize_language_code(
                    item
                )
                for item in raw.get(
                    "languages",
                    [language],
                )
            ]

            scores = self._parse_scores(
                raw.get(
                    "scores",
                    [],
                )
            )

            return LanguageDetectionResult(
                language=language,
                confidence=self._clamp(
                    confidence
                ),
                script=self.detect_script(
                    text
                ),
                method=DetectionMethod.INJECTED,
                languages=[
                    language
                    for language in languages
                    if language
                    != LanguageCode.UNKNOWN
                ],
                language_scores=scores,
            )

        if isinstance(
            raw,
            (list, tuple),
        ):
            scores = self._parse_scores(
                raw
            )

            if not scores:
                raise ValueError(
                    "Detector returned no language scores."
                )

            scores.sort(
                key=lambda item: item.score,
                reverse=True,
            )

            best = scores[0]

            return LanguageDetectionResult(
                language=best.language,
                confidence=self._clamp(
                    best.score
                ),
                script=self.detect_script(
                    text
                ),
                method=DetectionMethod.INJECTED,
                languages=[
                    item.language
                    for item in scores[
                        : self.config.max_languages
                    ]
                ],
                language_scores=scores,
            )

        raise TypeError(
            "Unsupported detector result."
        )

    # ------------------------------------------------------------------
    # Heuristic detection
    # ------------------------------------------------------------------

    def _heuristic_detect(
        self,
        text: str,
        script: ScriptCode,
    ) -> LanguageDetectionResult:
        tokens = self._tokenize(
            text
        )

        if not tokens:
            return LanguageDetectionResult(
                language=LanguageCode.UNKNOWN,
                confidence=0.0,
                script=script,
                method=DetectionMethod.FALLBACK,
                warnings=[
                    "No language-bearing tokens found."
                ],
            )

        scores: list[
            LanguageScore
        ] = []

        token_set = set(
            token.lower()
            for token in tokens
        )

        for language, markers in (
            self.LANGUAGE_MARKERS.items()
        ):
            score = 0.0
            evidence: list[str] = []

            for marker in markers:
                normalized_marker = (
                    marker.lower()
                )

                if normalized_marker in token_set:
                    score += 1.0
                    evidence.append(
                        marker
                    )

            if score > 0:
                scores.append(
                    LanguageScore(
                        language=language,
                        score=score,
                        evidence=evidence,
                    )
                )

        if not scores:
            script_language = (
                self._language_from_script(
                    script
                )
            )

            if script_language:
                return LanguageDetectionResult(
                    language=script_language,
                    confidence=0.45,
                    script=script,
                    method=DetectionMethod.SCRIPT,
                    languages=[
                        script_language
                    ],
                    warnings=[
                        "Language inferred primarily from script."
                    ],
                )

            return LanguageDetectionResult(
                language=LanguageCode.UNKNOWN,
                confidence=0.0,
                script=script,
                method=DetectionMethod.FALLBACK,
                warnings=[
                    "Unable to determine language reliably."
                ],
            )

        scores.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        best = scores[0]

        total_score = sum(
            item.score
            for item in scores
        )

        confidence = (
            best.score / total_score
            if total_score > 0
            else 0.0
        )

        # Heuristic detection should not pretend to be
        # highly certain.
        confidence = min(
            confidence,
            0.92,
        )

        if len(text) < 100:
            confidence = min(
                confidence,
                self.config.short_text_confidence_cap,
            )

        languages = [
            item.language
            for item in scores
            if (
                item.score
                / max(
                    best.score,
                    1.0,
                )
            )
            >= self.config.mixed_language_threshold
        ]

        result = LanguageDetectionResult(
            language=best.language,
            confidence=confidence,
            script=script,
            method=DetectionMethod.HEURISTIC,
            is_mixed=len(languages) > 1,
            languages=languages[
                : self.config.max_languages
            ],
            language_scores=scores[
                : self.config.max_languages
            ],
        )

        if confidence < (
            self.config.confidence_threshold
        ):
            result.warnings.append(
                "Language confidence is below the configured threshold."
            )

        return result

    # ------------------------------------------------------------------
    # Short text
    # ------------------------------------------------------------------

    def _detect_short_text(
        self,
        text: str,
    ) -> LanguageDetectionResult:
        script = self.detect_script(
            text
        )

        language = (
            self._language_from_script(
                script
            )
        )

        if language:
            return LanguageDetectionResult(
                language=language,
                confidence=0.35,
                script=script,
                method=DetectionMethod.SCRIPT,
                languages=[
                    language
                ],
                warnings=[
                    "Text is too short for reliable language detection."
                ],
            )

        return LanguageDetectionResult(
            language=LanguageCode.UNKNOWN,
            confidence=0.0,
            script=script,
            method=DetectionMethod.FALLBACK,
            warnings=[
                "Text is too short for reliable language detection."
            ],
        )

    # ------------------------------------------------------------------
    # Mixed-language detection
    # ------------------------------------------------------------------

    def _is_mixed_language(
        self,
        result: LanguageDetectionResult,
    ) -> bool:
        if len(
            result.languages
        ) <= 1:
            return False

        if not result.language_scores:
            return len(
                result.languages
            ) > 1

        scores = sorted(
            (
                score.score
                for score in result.language_scores
            ),
            reverse=True,
        )

        if len(scores) < 2:
            return False

        strongest = scores[0]

        second = scores[1]

        if strongest <= 0:
            return False

        return (
            second / strongest
            >= self.config.mixed_language_threshold
        )

    # ------------------------------------------------------------------
    # Script mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _language_from_script(
        script: ScriptCode,
    ) -> Optional[LanguageCode]:
        mapping = {
            ScriptCode.MALAYALAM: LanguageCode.MALAYALAM,
            ScriptCode.TAMIL: LanguageCode.TAMIL,
            ScriptCode.TELUGU: LanguageCode.TELUGU,
            ScriptCode.KANNADA: LanguageCode.KANNADA,
            ScriptCode.BENGALI: LanguageCode.BENGALI,
            ScriptCode.GURMUKHI: LanguageCode.PUNJABI,
            ScriptCode.GUJARATI: LanguageCode.GUJARATI,
            ScriptCode.DEVANAGARI: LanguageCode.HINDI,
            ScriptCode.CYRILLIC: LanguageCode.RUSSIAN,
            ScriptCode.HANGUL: LanguageCode.KOREAN,
            ScriptCode.HIRAGANA: LanguageCode.JAPANESE,
            ScriptCode.KATAKANA: LanguageCode.JAPANESE,
            ScriptCode.HAN: LanguageCode.CHINESE,
        }

        return mapping.get(
            script
        )

    @staticmethod
    def _script_confidence(
        text: str,
        script: ScriptCode,
    ) -> float:
        distribution = (
            LanguageDetector(
                config=LanguageDetectorConfig()
            ).script_distribution(
                text
            )
        )

        return round(
            distribution.get(
                script,
                0.0,
            ),
            4,
        )

    # ------------------------------------------------------------------
    # Character script
    # ------------------------------------------------------------------

    @staticmethod
    def _character_script(
        character: str,
    ) -> ScriptCode:
        codepoint = ord(
            character
        )

        # Malayalam
        if 0x0D00 <= codepoint <= 0x0D7F:
            return ScriptCode.MALAYALAM

        # Tamil
        if 0x0B80 <= codepoint <= 0x0BFF:
            return ScriptCode.TAMIL

        # Telugu
        if 0x0C00 <= codepoint <= 0x0C7F:
            return ScriptCode.TELUGU

        # Kannada
        if 0x0C80 <= codepoint <= 0x0CFF:
            return ScriptCode.KANNADA

        # Bengali
        if 0x0980 <= codepoint <= 0x09FF:
            return ScriptCode.BENGALI

        # Gurmukhi
        if 0x0A00 <= codepoint <= 0x0A7F:
            return ScriptCode.GURMUKHI

        # Gujarati
        if 0x0A80 <= codepoint <= 0x0AFF:
            return ScriptCode.GUJARATI

        # Devanagari
        if 0x0900 <= codepoint <= 0x097F:
            return ScriptCode.DEVANAGARI

        # Arabic
        if (
            0x0600 <= codepoint <= 0x06FF
            or 0x0750 <= codepoint <= 0x077F
            or 0x08A0 <= codepoint <= 0x08FF
        ):
            return ScriptCode.ARABIC

        # Cyrillic
        if (
            0x0400 <= codepoint <= 0x04FF
        ):
            return ScriptCode.CYRILLIC

        # CJK / Han
        if (
            0x4E00 <= codepoint <= 0x9FFF
        ):
            return ScriptCode.HAN

        # Hiragana
        if (
            0x3040 <= codepoint <= 0x309F
        ):
            return ScriptCode.HIRAGANA

        # Katakana
        if (
            0x30A0 <= codepoint <= 0x30FF
        ):
            return ScriptCode.KATAKANA

        # Hangul
        if (
            0xAC00 <= codepoint <= 0xD7AF
        ):
            return ScriptCode.HANGUL

        # Latin
        try:
            name = unicodedata.name(
                character
            )

            if "LATIN" in name:
                return ScriptCode.LATIN

        except ValueError:
            pass

        return ScriptCode.UNKNOWN

    # ------------------------------------------------------------------
    # Text preparation
    # ------------------------------------------------------------------

    def _prepare_text(
        self,
        text: str,
    ) -> str:
        text = text[: self.config.max_analysis_characters]

        if self.config.ignore_numbers:
            text = re.sub(
                r"\d+",
                " ",
                text,
            )

        if self.config.ignore_punctuation:
            text = "".join(
                character
                if (
                    character.isalpha()
                    or character.isspace()
                )
                else " "
                for character in text
            )

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:
        return re.findall(
            r"\b[\wÀ-ÿ\u0900-\u0DFF]+\b",
            text,
            flags=re.UNICODE,
        )

    # ------------------------------------------------------------------
    # Score parsing
    # ------------------------------------------------------------------

    def _parse_scores(
        self,
        raw_scores: Any,
    ) -> list[LanguageScore]:
        scores: list[
            LanguageScore
        ] = []

        if isinstance(
            raw_scores,
            dict,
        ):
            raw_scores = list(
                raw_scores.items()
            )

        if not isinstance(
            raw_scores,
            (list, tuple),
        ):
            return scores

        for item in raw_scores:
            if isinstance(
                item,
                LanguageScore,
            ):
                scores.append(item)
                continue

            if isinstance(
                item,
                dict,
            ):
                raw_language = item.get(
                    "language",
                    item.get(
                        "lang"
                    ),
                )

                raw_score = item.get(
                    "score",
                    item.get(
                        "confidence",
                        item.get(
                            "probability",
                            0.0,
                        ),
                    ),
                )

                if raw_language is None:
                    continue

                language = (
                    self._normalize_language_code(
                        raw_language
                    )
                )

                scores.append(
                    LanguageScore(
                        language=language,
                        score=self._clamp(
                            float(raw_score)
                        ),
                        evidence=list(
                            item.get(
                                "evidence",
                                [],
                            )
                        ),
                    )
                )

                continue

            if isinstance(
                item,
                (list, tuple),
            ) and len(item) >= 2:
                language = (
                    self._normalize_language_code(
                        item[0]
                    )
                )

                scores.append(
                    LanguageScore(
                        language=language,
                        score=self._clamp(
                            float(item[1])
                        ),
                    )
                )

        return [
            score
            for score in scores
            if score.language
            != LanguageCode.UNKNOWN
        ]

    # ------------------------------------------------------------------
    # Language code normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_language_code(
        value: Any,
    ) -> LanguageCode:
        if isinstance(
            value,
            LanguageCode,
        ):
            return value

        normalized = str(
            value
        ).strip().lower()

        aliases = {
            "eng": LanguageCode.ENGLISH,
            "english": LanguageCode.ENGLISH,
            "deu": LanguageCode.GERMAN,
            "ger": LanguageCode.GERMAN,
            "german": LanguageCode.GERMAN,
            "fra": LanguageCode.FRENCH,
            "fre": LanguageCode.FRENCH,
            "french": LanguageCode.FRENCH,
            "spa": LanguageCode.SPANISH,
            "spanish": LanguageCode.SPANISH,
            "ita": LanguageCode.ITALIAN,
            "italian": LanguageCode.ITALIAN,
            "por": LanguageCode.PORTUGUESE,
            "portuguese": LanguageCode.PORTUGUESE,
            "nld": LanguageCode.DUTCH,
            "dut": LanguageCode.DUTCH,
            "dutch": LanguageCode.DUTCH,
            "hin": LanguageCode.HINDI,
            "hindi": LanguageCode.HINDI,
            "mal": LanguageCode.MALAYALAM,
            "malayalam": LanguageCode.MALAYALAM,
            "tam": LanguageCode.TAMIL,
            "tamil": LanguageCode.TAMIL,
            "tel": LanguageCode.TELUGU,
            "telugu": LanguageCode.TELUGU,
            "kan": LanguageCode.KANNADA,
            "kannada": LanguageCode.KANNADA,
            "ben": LanguageCode.BENGALI,
            "bengali": LanguageCode.BENGALI,
            "mar": LanguageCode.MARATHI,
            "marathi": LanguageCode.MARATHI,
            "guj": LanguageCode.GUJARATI,
            "gujarati": LanguageCode.GUJARATI,
            "pan": LanguageCode.PUNJABI,
            "punjabi": LanguageCode.PUNJABI,
            "ara": LanguageCode.ARABIC,
            "arabic": LanguageCode.ARABIC,
            "rus": LanguageCode.RUSSIAN,
            "russian": LanguageCode.RUSSIAN,
            "ukr": LanguageCode.UKRAINIAN,
            "ukrainian": LanguageCode.UKRAINIAN,
            "zho": LanguageCode.CHINESE,
            "chi": LanguageCode.CHINESE,
            "chinese": LanguageCode.CHINESE,
            "jpn": LanguageCode.JAPANESE,
            "japanese": LanguageCode.JAPANESE,
            "kor": LanguageCode.KOREAN,
            "korean": LanguageCode.KOREAN,
            "nld": LanguageCode.DUTCH,
            "unknown": LanguageCode.UNKNOWN,
        }

        return aliases.get(
            normalized,
            LanguageCode.UNKNOWN,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )


# ======================================================================
# Convenience API
# ======================================================================


_default_detector = LanguageDetector()


def detect_language(
    text: str,
) -> LanguageDetectionResult:
    """
    Convenience language detection API.
    """

    return _default_detector.detect(
        text
    )


def detect_script(
    text: str,
) -> ScriptCode:
    """
    Convenience script detection API.
    """

    return _default_detector.detect_script(
        text
    )


__all__ = [
    "LanguageCode",
    "ScriptCode",
    "DetectionMethod",
    "LanguageScore",
    "LanguageDetectionResult",
    "LanguageDetectorConfig",
    "LanguageDetector",
    "detect_language",
    "detect_script",
]