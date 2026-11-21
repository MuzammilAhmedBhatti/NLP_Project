"""Local OCR for social-media screenshots."""

from functools import lru_cache

import cv2
import numpy as np


class OCRFailure(ValueError):
    pass


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def extract_text(image_bytes: bytes) -> str:
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
