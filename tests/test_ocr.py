import cv2
import numpy as np
import pytest

from modprecedent.ui.ocr import OCRFailure, extract_text


def png_with_text(text: str) -> bytes:
    image = np.full((180, 760, 3), 255, dtype=np.uint8)
    cv2.putText(image, text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    return cv2.imencode(".png", image)[1].tobytes()


def test_extracts_readable_image_text():
    assert "PUBLIC" in extract_text(png_with_text("PUBLIC INTEREST REPORT"))


def test_blank_image_has_clear_error():
    blank = cv2.imencode(".png", np.full((100, 100, 3), 255, dtype=np.uint8))[1].tobytes()
    with pytest.raises(OCRFailure, match="No readable text"):
        extract_text(blank)
