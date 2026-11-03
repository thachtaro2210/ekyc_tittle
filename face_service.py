import os
import threading
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort
import torch

from minifasnet import MiniFASNetV2

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MAX_DIM = 1080
MIN_FACE_SCORE = 0.6
SPOOF_CROP_SCALE = 2.7  # MiniFASNetV2 "2.7_80x80" expects the box enlarged 2.7x
SPOOF_REAL_THRESHOLD = 0.5
# Euclidean distance between L2-normalised VGGFace2 embeddings; lower = more similar.
MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.85"))


class FaceError(Exception):
    pass


class FaceVerifier:
    """Selfie liveness (MiniFASNet) + selfie-vs-CCCD-portrait matching (VGGFace2)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._detector = cv.FaceDetectorYN.create(
            str(WEIGHTS_DIR / "face_detection_yunet_2023mar.onnx"), "", (320, 320), MIN_FACE_SCORE
        )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._embedder = ort.InferenceSession(str(WEIGHTS_DIR / "vggface2_int8.onnx"), opts)

        self._spoof = MiniFASNetV2()
        state = torch.load(str(WEIGHTS_DIR / "2.7_80x80_MiniFASNetV2.pth"), map_location="cpu")
        self._spoof.load_state_dict(
            {k.replace("module.", "").replace(".model.", "."): v for k, v in state.items()}
        )
        self._spoof.eval()

    @staticmethod
    def _load(path: str) -> np.ndarray:
        img = cv.imread(path)
        if img is None:
            raise FaceError("Không đọc được ảnh")
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv.resize(img, (int(w * scale), int(h * scale)))
        return img

    def _largest_face(self, bgr: np.ndarray):
        """Returns (box x,y,w,h ints, eyes [(x,y),(x,y)]) of the largest face, or None."""
        self._detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda r: r[2] * r[3])
        box = tuple(int(v) for v in f[:4])
        eyes = sorted([(float(f[4]), float(f[5])), (float(f[6]), float(f[7]))])  # left, right by x
        return box, eyes

    @staticmethod
    def _spoof_crop(bgr: np.ndarray, box) -> np.ndarray:
        x, y, w, h = box
        src_h, src_w = bgr.shape[:2]
        scale = min((src_h - 1) / h, (src_w - 1) / w, SPOOF_CROP_SCALE)
        nw, nh = w * scale, h * scale
        left = min(max(x + w / 2 - nw / 2, 0), src_w - 1 - nw)
        top = min(max(y + h / 2 - nh / 2, 0), src_h - 1 - nh)
        crop = bgr[int(top):int(top + nh) + 1, int(left):int(left + nw) + 1]
        return cv.resize(crop, (80, 80))

    def _real_probability(self, bgr: np.ndarray, box) -> float:
        # Raw 0-255 BGR, as in the reference Silent-Face-Anti-Spoofing pipeline.
        x = torch.from_numpy(self._spoof_crop(bgr, box).transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self._spoof(x), dim=1)[0]
        return float(probs[1])  # classes: 0 = spoof, 1 = real, 2 = spoof

    def _embedding(self, bgr: np.ndarray, box, eyes) -> np.ndarray:
        x, y, w, h = box
        m = int(0.1 * max(w, h))
        x0, y0 = max(x - m, 0), max(y - m, 0)
        crop = bgr[y0:y + h + m, x0:x + w + m]

        (lx, ly), (rx, ry) = eyes
        angle = np.degrees(np.arctan2(ry - ly, rx - lx))
        centre = ((lx + rx) / 2 - x0, (ly + ry) / 2 - y0)
        crop = cv.warpAffine(
            crop,
            cv.getRotationMatrix2D(centre, angle, 1.0),
            (crop.shape[1], crop.shape[0]),
            borderMode=cv.BORDER_REPLICATE,
        )

        rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB)
        lab[..., 0] = cv.equalizeHist(lab[..., 0])
        rgb = cv.cvtColor(lab, cv.COLOR_LAB2RGB)

        face = (cv.resize(rgb, (160, 160)).astype(np.float32) - 127.5) / 128.0
        inp = face.transpose(2, 0, 1)[None]
        emb = self._embedder.run(None, {self._embedder.get_inputs()[0].name: inp})[0][0]
        return emb / np.linalg.norm(emb)

    def verify(self, selfie_paths: list[str], id_path: str) -> dict:
        with self._lock:
            best = None  # (area, bgr, box, eyes)
            real_scores = []
            for p in selfie_paths:
                img = self._load(p)
                found = self._largest_face(img)
                if not found:
                    continue
                box, eyes = found
                real_scores.append(self._real_probability(img, box))
                if best is None or box[2] * box[3] > best[0]:
                    best = (box[2] * box[3], img, box, eyes)
            if best is None:
                raise FaceError("Không phát hiện khuôn mặt trong ảnh selfie")

            spoof_score = float(np.mean(real_scores))
            liveness = spoof_score > SPOOF_REAL_THRESHOLD

            id_img = self._load(id_path)
            id_found = self._largest_face(id_img)
            if not id_found:
                raise FaceError("Không tìm thấy ảnh chân dung trên CCCD")

            selfie_emb = self._embedding(best[1], best[2], best[3])
            id_emb = self._embedding(id_img, *id_found)
            distance = float(np.linalg.norm(selfie_emb - id_emb))

            return {
                "liveness": liveness,
                "spoofScore": round(spoof_score, 3),
                "match": liveness and distance < MATCH_THRESHOLD,
                "distance": round(distance, 3),
                "threshold": MATCH_THRESHOLD,
                "embedding": selfie_emb.round(6).tolist(),
            }
