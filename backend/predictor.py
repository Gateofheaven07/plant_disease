"""Utilities for running disease prediction on plant leaf images.

This module no longer loads or runs a local TensorFlow/Keras model. Instead
it forwards prediction requests to a remote Gradio / Hugging Face Space via
the `gradio_client` library. The public API (exceptions and function
signatures) is preserved so callers (for example `api.py`) won't need
changes.
"""

from __future__ import annotations

import io
import json
import tempfile
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, UnidentifiedImageError
from leaf_validator import validate_leaf_image
import sys


def _get_client(space: str = "Taufik2307/plant_disease"):
    """Lazily create a gradio_client.Client and avoid printing unicode to
    consoles that can't encode it (Windows CP1252 issue).
    """
    try:
        from gradio_client import Client
    except Exception as exc:
        raise RuntimeError("gradio_client is required for remote predictions") from exc

    try:
        return Client(space)
    except UnicodeEncodeError:
        # Some Windows consoles cannot print the checkmark char gradio prints.
        # Temporarily redirect stdout to silence the print during init.
        old_stdout = sys.stdout
        try:
            sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
            return Client(space)
        finally:
            try:
                sys.stdout.close()
            except Exception:
                pass
            sys.stdout = old_stdout

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DISEASE_INFO_PATH = ARTIFACTS_DIR / "disease_info.json"



class PredictionError(ValueError):
    """Raised when the prediction pipeline cannot process the provided image."""


@lru_cache(maxsize=1)
def _load_disease_info() -> Dict[str, Dict[str, str]]:
    if not DISEASE_INFO_PATH.exists():
        raise FileNotFoundError(f"Disease info file not found at {DISEASE_INFO_PATH}")
    with DISEASE_INFO_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _call_remote_predict(file_path: str) -> dict:
    """Call the remote Gradio/HF Space predict API and return the raw dict.

    This expects the Space to return the JSON structure you provided earlier
    (keys: 'label', 'percentage', 'probabilities').
    """
    try:
        client = _get_client()
        result = client.predict(image=file_path, api_name="/predict")
    except Exception as exc:  # wrap any client/network errors
        raise PredictionError(f"Gagal memanggil model remote: {exc}") from exc

    if not isinstance(result, dict):
        raise PredictionError("Format response dari model remote tidak dikenali")
    return result


def predict_disease(image_path: str) -> Tuple[str, float]:
    """Run inference on an image stored on disk by forwarding to the HF Space.

    Returns (label, confidence_percentage).
    """
    try:
        with Image.open(image_path) as image:
            # Validate the image content before sending
            is_valid, error_message = validate_leaf_image(image)
            if not is_valid:
                raise PredictionError(error_message)
    except (FileNotFoundError, UnidentifiedImageError) as exc:
        raise PredictionError("File gambar tidak dapat diproses.") from exc

    result = _call_remote_predict(image_path)
    label = result.get("label")
    percentage = result.get("percentage")
    if label is None or percentage is None:
        raise PredictionError("Response model tidak mengandung label atau percentage")
    return label, float(percentage)


def predict_disease_from_bytes(data: bytes) -> Tuple[str, float]:
    """Run inference on raw image bytes (useful for API uploads).

    The bytes are validated and written to a temporary file which is then
    sent to the remote Space. The temporary file is removed afterwards.
    """
    if not data:
        raise PredictionError("File gambar kosong.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            is_valid, error_message = validate_leaf_image(image)
            if not is_valid:
                raise PredictionError(error_message)

            # Save a temporary PNG file to send to the remote API
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp_path = tmp.name
            try:
                image.convert("RGB").save(tmp_path, format="PNG")
                result = _call_remote_predict(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    except UnidentifiedImageError as exc:
        raise PredictionError("Format file tidak didukung.") from exc

    label = result.get("label")
    percentage = result.get("percentage")
    if label is None or percentage is None:
        raise PredictionError("Response model tidak mengandung label atau percentage")
    return label, float(percentage)


def get_disease_information(class_name: str) -> Dict[str, str]:
    """Return human readable metadata for a predicted disease."""
    info = _load_disease_info().get(class_name)
    if info:
        return info
    return {
        "nama": class_name.replace("_", " "),
        "deskripsi": "Informasi detail belum tersedia.",
        "pengobatan": "",
        "perawatan": "",
        "pencegahan": "",
    }
