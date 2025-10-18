# api.py
from __future__ import annotations

import os
import time
import tempfile
from http import HTTPStatus
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS
from gradio_client import handle_file
import sys


def _get_client(space: str = None):
    """Lazily create a gradio_client.Client. Avoid printing unicode to
    consoles that can't encode it (Windows CP1252 issue).
    """
    space = space or os.getenv("GRADIO_SPACE", "Taufik2307/plant_disease")
    try:
        from gradio_client import Client
    except Exception as exc:
        raise RuntimeError("gradio_client is required to call remote Space") from exc

    try:
        return Client(space)
    except UnicodeEncodeError:
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

# ---- Jika kamu punya modul lokal untuk info penyakit ----
# Pastikan file predictor.py ada dan berisi fungsi get_disease_information(label: str) -> dict
from predictor import get_disease_information  # type: ignore

# ---------------------------------------------------------
# Konfigurasi Aplikasi
# ---------------------------------------------------------
app = Flask(__name__)

# Izinkan frontend React di localhost:3000 (ubah/ tambah origin jika perlu)
CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:3000"]}},
    supports_credentials=True,
)

# Ganti dengan space/endpoint milikmu jika berbeda
# Contoh: "username/space_name" (Hugging Face Spaces) atau URL Gradio lain
GRADIO_SPACE = os.getenv("GRADIO_SPACE", "Taufik2307/plant_disease")

# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------
def _safe_remove(path: str, retries: int = 5, delay: float = 0.15) -> None:
    """Hapus file dengan retry kecil (Windows suka mengunci file sejenak)."""
    for _ in range(retries):
        try:
            os.remove(path)
            return
        except PermissionError:
            time.sleep(delay)
        except FileNotFoundError:
            return

def _normalize_result(result: Any) -> Dict[str, Any]:
    """
    Normalisasi keluaran gradio_client ke bentuk dict yang diharapkan frontend:
    {
      "prediction": <str>,
      "confidence": <float|int>,
      "probabilities": <dict|list>,
      "info": <dict>
    }
    """
    # Beberapa Spaces mengembalikan dict, ada juga yang list/tuple.
    if isinstance(result, dict):
        label = result.get("label") or result.get("prediction") or result.get("class") or ""
        confidence = result.get("percentage") or result.get("confidence") or result.get("score")
        probs = result.get("probabilities") or result.get("scores") or {}
    elif isinstance(result, (list, tuple)) and result:
        # Asumsi [label, confidence, probabilities] atau mirip
        label = str(result[0])
        confidence = result[1] if len(result) > 1 else None
        probs = result[2] if len(result) > 2 else {}
    else:
        label, confidence, probs = "", None, {}

    info = get_disease_information(label)  # dari predictor.py
    return {
        "prediction": label,
        "confidence": confidence,
        "probabilities": probs,
        "info": info,
    }

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/")
def root():
    return jsonify({"status": "ok", "message": "API is running", "space": GRADIO_SPACE})

@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), HTTPStatus.OK

@app.post("/predict")
def predict():
    # Validasi file
    if "file" not in request.files:
        return jsonify({"error": "File gambar tidak ditemukan pada request."}), HTTPStatus.BAD_REQUEST

    file_storage = request.files["file"]

    # Simpan ke file sementara lalu TUTUP agar tidak terkunci (penting di Windows)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_path = tmp.name
    tmp.close()

    # Simpan konten upload
    file_storage.save(temp_path)

    try:
        # Gunakan handle_file agar gradio_client mengunggah file path dengan benar
        # Ambil client secara lazy sehingga import tidak mencetak karakter unicode
        client = _get_client(GRADIO_SPACE)
        # Ganti argumen "image" dan "api_name" sesuai API di Space kamu jika berbeda.
        result = client.predict(image=handle_file(temp_path), api_name="/predict")

        payload = _normalize_result(result)
        return jsonify(payload), HTTPStatus.OK

    except Exception as e:
        # Kembalikan error terformat agar mudah dibaca di frontend
        return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST

    finally:
        _safe_remove(temp_path)

# ---------------------------------------------------------
# Entrypoint lokal
# ---------------------------------------------------------
if __name__ == "__main__":
    # Jalankan di http://localhost:8000 (sesuaikan dengan frontend)
    app.run(host="0.0.0.0", port=8000, debug=True)
