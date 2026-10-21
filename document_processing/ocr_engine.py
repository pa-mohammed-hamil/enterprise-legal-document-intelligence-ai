"""
OCR Engine

Optical Character Recognition for scanned documents and images.
"""

import logging
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


class OCREngine:
    """
    OCR engine for extracting text from images and scanned documents.
    
    Features:
    - Multi-language support
    - Image preprocessing for better accuracy
    - Table detection and extraction
    - Handwriting recognition
    """

    def __init__(
        self,
        languages: list[str] = None,
        tesseract_cmd: Optional[str] = None,
        preprocessing: bool = True,
    ):
        """
        Initialize OCR engine.

        Args:
            languages: List of language codes (e.g., ['eng', 'fra', 'deu'])
            tesseract_cmd: Path to tesseract executable
            preprocessing: Whether to apply image preprocessing
        """
        self.languages = languages or ["eng"]
        self.preprocessing = preprocessing

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text(
        self,
        image: Union[Image.Image, np.ndarray],
        psm: int = 3,
        confidence_threshold: float = 0.0,
    ) -> str:
        """
        Extract text from an image.

        Args:
            image: PIL Image or numpy array
            psm: Page segmentation mode (0-13)
                 3 = Fully automatic page segmentation (default)
                 6 = Assume a single uniform block of text
                 11 = Sparse text
            confidence_threshold: Minimum confidence to include text (0-100)

        Returns:
            Extracted text
        """
        try:
            # Convert to PIL Image if numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Preprocess image for better OCR
            if self.preprocessing:
                image = self._preprocess_image(image)

            # Configure OCR
            lang = "+".join(self.languages)
            config = f"--psm {psm} --oem 3"

            # Extract text with confidence
            if confidence_threshold > 0:
                data = pytesseract.image_to_data(
                    image, lang=lang, config=config, output_type=pytesseract.Output.DICT
                )
                
                # Filter by confidence
                text_parts = []
                for i, conf in enumerate(data["conf"]):
                    if int(conf) >= confidence_threshold:
                        text = data["text"][i]
                        if text.strip():
                            text_parts.append(text)
                
                text = " ".join(text_parts)
            else:
                text = pytesseract.image_to_string(image, lang=lang, config=config)

            return text.strip()

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""

    def extract_text_with_confidence(
        self, image: Union[Image.Image, np.ndarray]
    ) -> list[dict]:
        """
        Extract text with word-level confidence scores.

        Args:
            image: PIL Image or numpy array

        Returns:
            List of dictionaries with text and confidence:
            [{"text": str, "confidence": float, "bbox": tuple}, ...]
        """
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            if self.preprocessing:
                image = self._preprocess_image(image)

            lang = "+".join(self.languages)
            data = pytesseract.image_to_data(
                image, lang=lang, output_type=pytesseract.Output.DICT
            )

            results = []
            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                if text:
                    results.append({
                        "text": text,
                        "confidence": float(data["conf"][i]),
                        "bbox": (
                            data["left"][i],
                            data["top"][i],
                            data["width"][i],
                            data["height"][i],
                        ),
                    })

            return results

        except Exception as e:
            logger.error(f"OCR with confidence failed: {e}")
            return []

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR accuracy.

        Techniques:
        - Grayscale conversion
        - Denoising
        - Contrast enhancement
        - Binarization
        """
        try:
            # Convert PIL to OpenCV
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # Binarization with Otsu's method
            _, binary = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Convert back to PIL
            return Image.fromarray(binary)

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}, using original")
            return image

    def detect_tables(self, image: Union[Image.Image, np.ndarray]) -> list[dict]:
        """
        Detect tables in an image.

        Args:
            image: PIL Image or numpy array

        Returns:
            List of detected table regions with bounding boxes
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image

            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Threshold
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

            horizontal_lines = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN, horizontal_kernel
            )
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

            # Combine lines
            table_mask = cv2.add(horizontal_lines, vertical_lines)

            # Find contours
            contours, _ = cv2.findContours(
                table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            tables = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by size (minimum table size)
                if w > 100 and h > 100:
                    tables.append({
                        "bbox": (x, y, w, h),
                        "area": w * h,
                    })

            return sorted(tables, key=lambda t: t["area"], reverse=True)

        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            return []

    def extract_table_text(
        self, image: Union[Image.Image, np.ndarray], table_bbox: tuple
    ) -> list[list[str]]:
        """
        Extract text from a detected table.

        Args:
            image: PIL Image or numpy array
            table_bbox: Table bounding box (x, y, w, h)

        Returns:
            2D list representing table cells
        """
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image

            x, y, w, h = table_bbox
            table_region = img_array[y : y + h, x : x + w]

            # OCR on table region
            table_image = Image.fromarray(table_region)
            text = self.extract_text(table_image, psm=6)

            # Simple row detection (split by lines)
            rows = [line.strip() for line in text.split("\n") if line.strip()]

            # Simple column detection (split by multiple spaces)
            table_data = []
            for row in rows:
                cells = [cell.strip() for cell in row.split("  ") if cell.strip()]
                if cells:
                    table_data.append(cells)

            return table_data

        except Exception as e:
            logger.error(f"Table text extraction failed: {e}")
            return []

    def get_supported_languages(self) -> list[str]:
        """
        Get list of languages supported by Tesseract installation.

        Returns:
            List of language codes
        """
        try:
            langs = pytesseract.get_languages()
            return langs
        except Exception as e:
            logger.error(f"Could not get supported languages: {e}")
            return ["eng"]

    def detect_orientation(self, image: Union[Image.Image, np.ndarray]) -> float:
        """
        Detect text orientation angle.

        Args:
            image: PIL Image or numpy array

        Returns:
            Orientation angle in degrees
        """
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            osd = pytesseract.image_to_osd(image)
            
            # Parse orientation from OSD output
            for line in osd.split("\n"):
                if "Rotate:" in line:
                    angle = float(line.split(":")[1].strip())
                    return angle

            return 0.0

        except Exception as e:
            logger.warning(f"Orientation detection failed: {e}")
            return 0.0

    def rotate_to_correct_orientation(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Image.Image:
        """
        Automatically rotate image to correct orientation.

        Args:
            image: PIL Image or numpy array

        Returns:
            Rotated PIL Image
        """
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            angle = self.detect_orientation(image)
            
            if angle != 0:
                logger.info(f"Rotating image by {angle} degrees")
                return image.rotate(-angle, expand=True)

            return image

        except Exception as e:
            logger.warning(f"Auto-rotation failed: {e}")
            return image if isinstance(image, Image.Image) else Image.fromarray(image)
