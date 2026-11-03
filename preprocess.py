import cv2 as cv
import numpy as np


def enhance_image_for_ocr(img: np.ndarray) -> np.ndarray:
    """CLAHE + denoise + unsharp mask; returns 3-channel RGB for VietOCR."""
    if len(img.shape) == 3:
        denoised = cv.fastNlMeansDenoisingColored(
            img, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21
        )
        gray = cv.cvtColor(denoised, cv.COLOR_RGB2GRAY)
    else:
        gray = cv.fastNlMeansDenoising(img, None, h=6, templateWindowSize=7, searchWindowSize=21)

    enhanced = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gaussian = cv.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
    sharpened = np.clip(cv.addWeighted(enhanced, 1.5, gaussian, -0.5, 0), 0, 255).astype(np.uint8)
    return cv.cvtColor(sharpened, cv.COLOR_GRAY2RGB)
