import logging
import os
import tempfile

from flask import Flask, jsonify, request

from face_service import FaceError, FaceVerifier
from ocr_service import CccdOcr, OcrError

app = Flask(__name__)
app.logger.setLevel(logging.INFO)  # Flask only shows WARNING+ by default
ocr = CccdOcr()  # load models once at startup
face = FaceVerifier()


def _describe_result(result: dict) -> str:
    """Log the shape of an OCR result, never the personal values."""
    return ", ".join(f"{key}={len(str(value))} chars" for key, value in result.items())


def _run(field: str, extract):
    file = request.files.get(field)
    if not file or not (file.mimetype or "").startswith("image/"):
        return jsonify({"error": f"Thiếu file ảnh '{field}'"}), 400

    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        file.save(path)
        result = extract(path)
        app.logger.info("OCR %s ok: %s", field, _describe_result(result))
        return jsonify({"result": result})
    except OcrError as e:
        app.logger.warning("OCR %s rejected: %s", field, e)
        return jsonify({"error": str(e)}), 422
    except Exception:
        app.logger.exception("OCR failed")
        return jsonify({"error": "Lỗi xử lý ảnh"}), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.post("/api/ocr/front")
def front():
    return _run("frontID", ocr.extract_front)


@app.post("/api/ocr/back")
def back():
    return _run("backID", ocr.extract_back)


@app.post("/api/face/verify")
def face_verify():
    selfies = [f for f in request.files.getlist("selfie") if (f.mimetype or "").startswith("image/")]
    id_file = request.files.get("frontID")
    if not selfies or not id_file or not (id_file.mimetype or "").startswith("image/"):
        return jsonify({"error": "Cần ít nhất 1 ảnh 'selfie' và ảnh 'frontID'"}), 400

    paths = []
    try:
        for f in [*selfies, id_file]:
            fd, path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            paths.append(path)
            f.save(path)
        result = face.verify(paths[:-1], paths[-1])
        app.logger.info(
            "Face verify ok: liveness=%s match=%s distance=%s spoofScore=%s",
            result["liveness"], result["match"], result["distance"], result["spoofScore"],
        )
        return jsonify({"result": result})
    except FaceError as e:
        app.logger.warning("Face verify rejected: %s", e)
        return jsonify({"error": str(e)}), 422
    except Exception:
        app.logger.exception("Face verify failed")
        return jsonify({"error": "Lỗi xử lý ảnh"}), 500
    finally:
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass


@app.post("/api/face/pose")
def face_pose():
    image = request.files.get("image")
    expected = request.form.get("expected", "")
    if not image or not (image.mimetype or "").startswith("image/"):
        return jsonify({"error": "Thiếu file ảnh 'image'"}), 400
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        image.save(path)
        result = face.pose(path, expected)
        app.logger.info(
            "Face pose ok: expected=%s pose=%s matches=%s", expected, result["pose"], result["matches"]
        )
        return jsonify({"result": result})
    except FaceError as e:
        app.logger.warning("Face pose rejected: %s", e)
        return jsonify({"error": str(e)}), 422
    except Exception:
        app.logger.exception("Face pose failed")
        return jsonify({"error": "Lỗi xử lý ảnh"}), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.get("/health")
def health():
    return jsonify({"status": "ok"})
