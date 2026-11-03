import os
import tempfile

from flask import Flask, jsonify, request

from face_service import FaceError, FaceVerifier
from ocr_service import CccdOcr, OcrError

app = Flask(__name__)
ocr = CccdOcr()  # load models once at startup
face = FaceVerifier()


def _run(field: str, extract):
    file = request.files.get(field)
    if not file or not (file.mimetype or "").startswith("image/"):
        return jsonify({"error": f"Thiếu file ảnh '{field}'"}), 400

    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        file.save(path)
        return jsonify({"result": extract(path)})
    except OcrError as e:
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
        return jsonify({"result": face.verify(paths[:-1], paths[-1])})
    except FaceError as e:
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


@app.get("/health")
def health():
    return jsonify({"status": "ok"})
