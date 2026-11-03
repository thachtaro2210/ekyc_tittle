# ekyc_tittle

Vietnamese CCCD OCR + selfie liveness / face matching service (Flask).

| Endpoint | Input (form-data) | Output |
|---|---|---|
| `POST /api/ocr/front` | `frontID` | idNumber, fullName, dob, gender, nationality, birthplace, address |
| `POST /api/ocr/back` | `backID` | address, issueDate, expiryDate, placeOfBirth |
| `POST /api/face/verify` | `selfie` (1+), `frontID` | liveness, spoofScore, match, distance, embedding |
| `GET /health` | | status |

Run: `pip install -r requirements.txt && python -m flask --app app run --port 8000`
or `docker build -t kyc-ocr . && docker run -p 8000:8000 kyc-ocr`.
First run downloads the VietOCR `vgg_transformer` weights.

Derived from the eKYC service by Tinh Pham (HDBank AMC), used with permission.
