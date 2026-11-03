import gc
import re
import threading
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import ultralytics
import yolov5
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

import geometry
from preprocess import enhance_image_for_ocr

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MAX_DIM = 1080
BACK_CONF = 0.6
CONTENT_CONF = 0.75

BACK_NEW_CLASSES = {
    0: "Date of expirty", 1: "Date of issue", 2: "Place", 3: "Place of birth",
    4: "address_1", 5: "address_2", 6: "bottom_left", 7: "bottom_right",
    8: "cdate_of_birth", 9: "cdate_of_expiry", 10: "cdate_of_issue",
    11: "cplace_of_birth", 12: "date_of_expiry", 13: "date_of_issue",
    14: "place", 15: "place_of_birth", 16: "top_left", 17: "top_right",
}
BACK_OLD_ISSUE_DATE = 0


class OcrError(Exception):
    pass


class CccdOcr:
    """Extracts text fields from a Vietnamese CCCD (front: old+new, back: old+new chip)."""

    def __init__(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._corner_front = yolov5.load(str(WEIGHTS_DIR / "corner_back.pt"))
        self._content_front = yolov5.load(str(WEIGHTS_DIR / "content.pt"))
        self._content_front.conf = CONTENT_CONF
        self._back_new = ultralytics.YOLO(str(WEIGHTS_DIR / "new_content_back.onnx"), task="detect")
        self._back_old = ultralytics.YOLO(str(WEIGHTS_DIR / "content_back.onnx"), task="detect")

        cfg = Cfg.load_config_from_name("vgg_transformer")
        cfg["cnn"]["pretrained"] = False
        cfg["device"] = device
        cfg["predictor"]["beamsearch"] = True
        self._reader = Predictor(cfg)

    def _read(self, crop: np.ndarray) -> str:
        return self._reader.predict(Image.fromarray(enhance_image_for_ocr(crop))).strip()

    @staticmethod
    def _load(path: str) -> np.ndarray:
        img = cv.imread(path)
        if img is None:
            raise OcrError("Không đọc được ảnh")
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv.resize(img, (int(w * scale), int(h * scale)))
        return img

    def _align_front(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        preds = self._corner_front(path).pred[0]
        if preds is None or preds.size(0) < 4:
            raise OcrError("Không phát hiện đủ 4 góc CCCD")
        ordered = geometry.class_Order(preds[:, :4].tolist(), preds[:, 5].tolist())
        pts = list(map(geometry.get_center_point, ordered))
        pts = [pts[0], pts[1], (pts[2][0], pts[2][1] + 30), (pts[3][0], pts[3][1] + 30)]
        return Image.fromarray(geometry.four_point_transform(img, np.asarray(pts)))

    def extract_front(self, path: str) -> dict:
        with self._lock:
            try:
                aligned = self._align_front(path)
                preds = self._content_front(aligned).pred[0]
                if preds is None or preds.size(0) == 0:
                    raise OcrError("Không phát hiện nội dung mặt trước")

                classes_raw = preds[:, 5].tolist()
                boxes, classes = geometry.non_max_suppression_fast(
                    np.array(preds[:, :4].tolist()), classes_raw, 0.7
                )
                boxes = geometry.class_Order(boxes, classes)

                img = np.array(aligned)
                fields = []
                for i, box in enumerate(boxes):
                    left, top, right, bottom = map(int, box)
                    if 5 < i < 9:
                        right += 100
                    if i > 0:  # lowest class id (box 0) is not a text field; skipped as in the source service
                        fields.append(self._read(img[top:bottom, left:right]))

                if 7 in classes_raw and len(fields) >= 9:
                    fields = fields[:6] + [fields[6] + ", " + fields[7]] + [fields[8]]

                names = (
                    ["idNumber", "fullName", "dob", "gender", "nationality", "birthplace", "address"]
                    if len(fields) >= 7
                    else ["idNumber", "fullName", "gender", "nationality", "dob"]
                )
                return dict(zip(names, fields))
            finally:
                gc.collect()

    def extract_back(self, path: str) -> dict:
        with self._lock:
            try:
                img = self._load(path)

                found = {}
                for box in self._back_new(img, verbose=False)[0].boxes:
                    if box.conf[0].item() < BACK_CONF:
                        continue
                    label = BACK_NEW_CLASSES.get(int(box.cls[0].item()))
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    text = self._read(img[y1:y2, x1:x2])
                    if label and len(text) >= 3:
                        found[label] = text

                if found and ("cdate_of_issue" in found or "Date of issue" in found):
                    def val(key: str) -> str:
                        return found.get(key, "").split(":", 1)[-1].strip()

                    return {
                        "version": "new",
                        "address": f"{val('address_1')} {val('address_2')}".strip(),
                        "issueDate": val("cdate_of_issue"),
                        "expiryDate": val("cdate_of_expiry"),
                        "placeOfBirth": val("cplace_of_birth"),
                    }

                for box in self._back_old(img, verbose=False)[0].boxes:
                    if box.conf[0].item() < BACK_CONF or int(box.cls[0].item()) != BACK_OLD_ISSUE_DATE:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    text = self._read(img[y1:y2, x1:x2])
                    m = re.search(r"(\d{2}[/\-.]\d{2}[/\-.]\d{4})", text)
                    return {"version": "old", "issueDate": m.group(1) if m else text}

                raise OcrError("Không phát hiện được thông tin mặt sau")
            finally:
                gc.collect()
