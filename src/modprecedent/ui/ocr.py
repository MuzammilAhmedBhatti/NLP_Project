"""Local OCR for social-media screenshots."""

import io
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from ..config import settings


class OCRFailure(ValueError):
    pass


@lru_cache(maxsize=1)
def _engine():
    """Create and cache the RapidOCR engine."""
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def extract_text(image_bytes: bytes) -> str:
    """Extract readable text from an uploaded image or raise a clear error."""
    if not image_bytes:
        raise OCRFailure("Choose an image containing readable text.")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OCRFailure("The uploaded file is not a readable image.")
    result, _ = _engine()(image)
    text = "\n".join(item[1].strip() for item in (result or []) if item[1].strip())
    if len("".join(text.split())) < 3:
        raise OCRFailure("No readable text could be extracted. Try a sharper image.")
    return text


@lru_cache(maxsize=1)
def _vision_engine():
    """Create and cache the optional local BLIP captioning pipeline."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    cache_dir = settings.model_cache_dir or None
    processor = BlipProcessor.from_pretrained(
        settings.vision_model, cache_dir=cache_dir, local_files_only=True
    )
    model = BlipForConditionalGeneration.from_pretrained(
        settings.vision_model, cache_dir=cache_dir, local_files_only=True
    ).eval()
    return processor, model


def describe_image(image_bytes: bytes) -> str:
    """Generate a local visual caption to complement, not replace, OCR."""
    if not settings.vision_enabled:
        return ""
    import torch

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processor, model = _vision_engine()
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=45)
    return processor.decode(output[0], skip_special_tokens=True).strip()


def extract_content(image_bytes: bytes) -> str:
    """Extract readable text and add a visual description when available."""
    text = extract_text(image_bytes)
    try:
        visual = describe_image(image_bytes)
    except Exception:  # noqa: BLE001 - OCR remains usable if the optional vision model fails
        visual = ""
    if visual:
        return f"OCR text:\n{text}\n\nVisual description:\n{visual}"
    return text
