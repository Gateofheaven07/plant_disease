"""Flask API for the plant leaf disease detection service."""

from __future__ import annotations

from http import HTTPStatus

from flask import Flask, jsonify, request
from flask_cors import CORS

from predictor import (
    PredictionError,
    get_disease_information,
    predict_disease_from_bytes,
)

app = Flask(__name__)
CORS(app)


@app.get("/")
def health() -> tuple[dict, int]:
    """Simple health-check endpoint."""
    return {"status": "ok"}, HTTPStatus.OK


@app.post("/predict")
def predict() -> tuple[dict, int]:
    """Handle incoming image uploads and return model predictions."""
    if "file" not in request.files:
        return {"error": "File gambar tidak ditemukan pada request."}, HTTPStatus.BAD_REQUEST

    file_storage = request.files["file"]
    data = file_storage.read()

    try:
        label, confidence = predict_disease_from_bytes(data)
    except PredictionError as error:
        return {"error": str(error)}, HTTPStatus.BAD_REQUEST

    info = get_disease_information(label)

    response_payload = {
        "prediction": label,
        "confidence": round(confidence, 2),
        "info": info,
    }
    return jsonify(response_payload), HTTPStatus.OK


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=8000, debug=False)
