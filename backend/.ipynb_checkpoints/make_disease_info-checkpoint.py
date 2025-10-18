import io, json, numpy as np
from pathlib import Path
from PIL import Image
from tensorflow import keras
import joblib

# ---- base paths: relatif ke file ini, bukan CWD ----
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATHS = [BASE_DIR / "model.keras", BASE_DIR / "model.h5"]
RF_PATH = BASE_DIR / "rf.joblib"
CLASS_IDX_PATH = ARTIFACTS_DIR / "class_indices.json"
DISEASE_INFO_PATH = ARTIFACTS_DIR / "disease_info.json"

class Predictor:
    def __init__(self):
        # CNN
        self.cnn = None
        for p in MODEL_PATHS:
            if p.exists():
                self.cnn = keras.models.load_model(str(p))
                self.model_path = p
                break
        if self.cnn is None:
            raise FileNotFoundError("Tidak menemukan model.keras maupun model.h5 di folder project.")

        # RandomForest (opsional)
        try:
            self.rf = joblib.load(RF_PATH)
        except Exception:
            self.rf = None

        # artifacts
        if not CLASS_IDX_PATH.exists():
            raise FileNotFoundError(f"Tidak menemukan {CLASS_IDX_PATH}. Jalankan train.py terlebih dahulu.")
        with CLASS_IDX_PATH.open(encoding="utf-8") as f:
            self.class_idx = {int(k): v for k, v in json.load(f).items()}

        # buat disease_info.json kalau belum ada
        if not DISEASE_INFO_PATH.exists():
            DISEASE_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
            template = {
                name: {
                    "nama": name.replace("_", " "),
                    "deskripsi": "",
                    "pengobatan": "",
                    "perawatan": "",
                    "pencegahan": "",
                }
                for name in self.class_idx.values()
            }
            DISEASE_INFO_PATH.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")

        with DISEASE_INFO_PATH.open(encoding="utf-8") as f:
            self.info = json.load(f)

        # embedding extractor
        self.emb_extractor = keras.Model(self.cnn.input, self.cnn.get_layer("gap").output)

        # infer image size (H, W)
        ih, iw = self.cnn.input_shape[1], self.cnn.input_shape[2]
        self.img_size = (ih, iw)

    def _preprocess(self, pil_img: Image.Image):
        pil_img = pil_img.convert("RGB").resize(self.img_size)
        x = np.expand_dims(np.array(pil_img, dtype=np.float32), 0)
        return x

    def _rf_proba_full(self, emb, num_classes):
        proba = self.rf.predict_proba(emb)
        if proba.shape[1] == num_classes:
            return proba
        full = np.zeros((proba.shape[0], num_classes), dtype=proba.dtype)
        full[:, self.rf.classes_] = proba
        return full

    def predict_pil(self, pil_img: Image.Image, use_ensemble: bool = True, alpha: float = 0.6):
        x = self._preprocess(pil_img)
        p_cnn = self.cnn.predict(x, verbose=0)
        probs = p_cnn
        if use_ensemble and self.rf is not None:
            emb = self.emb_extractor.predict(x, verbose=0)
            p_rf = self._rf_proba_full(emb, p_cnn.shape[1])
            probs = alpha * p_cnn + (1 - alpha) * p_rf
        cls_id = int(np.argmax(probs))
        cls_name = self.class_idx.get(cls_id, str(cls_id))
        return {
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": float(np.max(probs)),
            "info": self.info.get(cls_name, {}),
        }

    def predict_bytes(self, data: bytes, use_ensemble: bool = True, alpha: float = 0.6):
        pil = Image.open(io.BytesIO(data))
        return self.predict_pil(pil, use_ensemble=use_ensemble, alpha=alpha)

    def predict_file(self, path: str, use_ensemble: bool = True, alpha: float = 0.6):
        pil = Image.open(path)
        return self.predict_pil(pil, use_ensemble=use_ensemble, alpha=alpha)
