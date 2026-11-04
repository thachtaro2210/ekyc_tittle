import os
import threading
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort
import torch
from facenet_pytorch import MTCNN

from minifasnet import MiniFASNetV2

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MAX_DIM = 1080
MIN_FACE_SCORE = 0.9
EMBED_MARGIN = 0.25  # wider than the detector box: keeps hairline and chin, which helps selfie-vs-ID matching
SPOOF_CROP_SCALE = 2.7  # MiniFASNetV2 "2.7_80x80" expects the box enlarged 2.7x
SPOOF_REAL_THRESHOLD = 0.5
# Euclidean distance between L2-normalised VGGFace2 embeddings; lower = more similar.
MATCH_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "1.05"))


# Head-pose challenge thresholds, in units of inter-eye distance (yaw) and of the eye-to-mouth span (pitch).
POSE_CENTER_YAW = 0.15
POSE_TURN_YAW = 0.25
POSE_UP_DELTA = 0.12  # nose ratio must drop this far below the user's own frontal ratio
POSE_NAMES = ("center", "left", "right", "up")


class FaceError(Exception):
    pass


def _pose_metrics(lm) -> tuple[float, float]:
    """yaw: nose offset from the eye midpoint / eye distance (+ = nose toward image right, i.e. user turned to their left).
    pitch: nose position between eye line and mouth line (0 = eyes, 1 = mouth); falls when looking up."""
    (lx, ly), (rx, ry), (nx, ny) = lm[0], lm[1], lm[2]
    mouth_y = (lm[3][1] + lm[4][1]) / 2
    eye_mid_x, eye_mid_y = (lx + rx) / 2, (ly + ry) / 2
    eye_dist = max(abs(rx - lx), 1e-6)
    span = max(mouth_y - eye_mid_y, 1e-6)
    return float((nx - eye_mid_x) / eye_dist), float((ny - eye_mid_y) / span)


def classify_pose(yaw: float, pitch: float, base_pitch: float | None) -> str | None:
    """Maps metrics to one of POSE_NAMES. `base_pitch` is the pitch of the user's own frontal capture."""
    if abs(yaw) >= POSE_TURN_YAW:
        return "left" if yaw > 0 else "right"  # unmirrored capture: user turning to their left shows the nose on image right
    if abs(yaw) > POSE_CENTER_YAW:
        return None  # between frontal and turned: ask the user to commit
    if base_pitch is not None and pitch <= base_pitch - POSE_UP_DELTA:
        return "up"
    return "center"


class FaceVerifier:
    """Selfie liveness (MiniFASNet) + selfie-vs-CCCD-portrait matching (VGGFace2)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._detector = MTCNN(keep_all=True, device="cpu")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        # fp32 separates same/different people far better than int8 on ID-card portraits.
        fp32 = WEIGHTS_DIR / "vggface2.onnx"
        self._embedder = ort.InferenceSession(
            str(fp32 if fp32.exists() else WEIGHTS_DIR / "vggface2_int8.onnx"), opts
        )

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
        """Returns (box x,y,w,h ints, eyes [(x,y),(x,y)], 5 landmarks) of the largest face, or None."""
        boxes, probs, landmarks = self._detector.detect(cv.cvtColor(bgr, cv.COLOR_BGR2RGB), landmarks=True)
        if boxes is None:
            return None
        faces = [(b, lm) for b, p, lm in zip(boxes, probs, landmarks) if p is not None and p >= MIN_FACE_SCORE]
        if not faces:
            return None
        b, lm = max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
        x, y = max(int(b[0]), 0), max(int(b[1]), 0)
        box = (x, y, int(b[2]) - x, int(b[3]) - y)
        eyes = sorted([(float(lm[0][0]), float(lm[0][1])), (float(lm[1][0]), float(lm[1][1]))])  # left, right by x
        return box, eyes, lm

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
        m = int(EMBED_MARGIN * max(w, h))
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

        face = (cv.resize(rgb, (160, 160)).astype(np.float32) - 127.5) / 128.0
        inp = face.transpose(2, 0, 1)[None]
        emb = self._embedder.run(None, {self._embedder.get_inputs()[0].name: inp})[0][0]
        return emb / np.linalg.norm(emb)

    def pose(self, path: str, expected: str, base_pitch: float | None = None, mirrored: bool = False) -> dict:
        """Head-pose challenge step: does this frame show `expected` (one of POSE_NAMES)?"""
        if expected not in POSE_NAMES:
            raise FaceError("Hướng khuôn mặt không hợp lệ")
        with self._lock:
            found = self._largest_face(self._load(path))
            if not found:
                return {"faceFound": False, "pose": None, "matches": False}
            yaw, pitch = _pose_metrics(found[2])
            if mirrored:
                yaw = -yaw
            pose = classify_pose(yaw, pitch, base_pitch)
            return {
                "faceFound": True,
                "pose": pose,
                "matches": pose == expected,
                "yaw": round(yaw, 3),
                "pitch": round(pitch, 3),
            }

    def verify(self, selfie_paths: list[str], id_path: str) -> dict:
        with self._lock:
            best = None  # (area, bgr, box, eyes)
            real_scores = []
            for p in selfie_paths:
                img = self._load(p)
                found = self._largest_face(img)
                if not found:
                    continue
                box, eyes, _ = found
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
            id_emb = self._embedding(id_img, *id_found[:2])
            distance = float(np.linalg.norm(selfie_emb - id_emb))

            return {
                "liveness": liveness,
                "spoofScore": round(spoof_score, 3),
                "match": liveness and distance < MATCH_THRESHOLD,
                "distance": round(distance, 3),
                "threshold": MATCH_THRESHOLD,
                "embedding": selfie_emb.round(6).tolist(),
            }
