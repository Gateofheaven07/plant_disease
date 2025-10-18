#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib


# In[3]:


# Konfigurasi
# =========================
DATASET_CANDIDATES = [
    os.environ.get("DATASET_NAME", "DATASET_DAUN"),
    "DATASET-DAUN",
]
IMG_SIZE = (160, 160)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
SEED = 1337
EPOCHS_STAGE1 = int(os.environ.get("EPOCHS_STAGE1", 10))
EPOCHS_STAGE2 = int(os.environ.get("EPOCHS_STAGE2", 10))

ARTIFACTS_DIR = Path("artifacts"); ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = Path("model.keras")  # <-- pakai format .keras (lebih aman)
RF_PATH = Path("rf.joblib")


# In[4]:


# Util: cari folder dataset dengan aman
# =========================
def find_dataset_dir(names):
    start = Path.cwd()
    for base in (start, *start.parents):
        for name in names:
            cand = base / name
            if cand.is_dir():
                return cand.resolve()
    for name in names:
        for cand in start.rglob(name):
            if cand.is_dir():
                return cand.resolve()
    raise FileNotFoundError(
        "Tidak menemukan folder dataset. Dicari kandidat: "
        f"{names}\nCWD: {start}\n"
        "Solusi: pindahkan folder dataset ke CWD ini, atau set env DATASET_NAME."
    )


# In[5]:


# Muat dataset + validasi
# =========================
print("Mencari folder dataset…")
DATA_DIR = find_dataset_dir(DATASET_CANDIDATES)
print("DATA_DIR =", DATA_DIR)

subdirs = [p for p in DATA_DIR.iterdir() if p.is_dir()]
if not subdirs:
    raise RuntimeError(f"Folder {DATA_DIR} tidak berisi subfolder kelas.")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
num_imgs = sum(1 for p in DATA_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS)
if num_imgs == 0:
    raise RuntimeError(f"Tidak menemukan file gambar di {DATA_DIR}.")

print(f"Jumlah kelas: {len(subdirs)}; Perkiraan gambar: {num_imgs}")

abs_data_dir = str(DATA_DIR)

train_ds = keras.utils.image_dataset_from_directory(
    abs_data_dir, labels="inferred", label_mode="int",
    validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
val_ds = keras.utils.image_dataset_from_directory(
    abs_data_dir, labels="inferred", label_mode="int",
    validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Kelas:", class_names)

with (ARTIFACTS_DIR / "class_indices.json").open("w", encoding="utf-8") as f:
    json.dump({i: n for i, n in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)


# In[6]:


# Model: Augmentasi + MobileNetV2
# =========================
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name="augmentation",
)

base = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base.trainable = False  # Stage 1: beku

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)

# ⬇️ GANTI preprocessing agar kompatibel save/load
# x = keras.applications.mobilenet_v2.preprocess_input(x)
x = layers.Rescaling(1.0/127.5, offset=-1.0)(x)

x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax", name="preds")(x)
model = keras.Model(inputs, outputs, name="leaf_disease_cnn")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# In[7]:


# Class weights (atasi ketidakseimbangan)
# =========================
labels_collected = []
tmp_ds = keras.utils.image_dataset_from_directory(
    abs_data_dir, labels="inferred", label_mode="int",
    validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE,
)
for _, yb in tmp_ds:
    labels_collected.append(yb.numpy())
y_train_all = np.concatenate(labels_collected)
cw_vals = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train_all)
class_weight = {i: float(w) for i, w in enumerate(cw_vals)}
print("Class weights:", class_weight)


# In[8]:


# Stage 1 (feature extractor frozen)
# =========================
ckpt = keras.callbacks.ModelCheckpoint(
    str(MODEL_PATH), monitor="val_accuracy", mode="max",
    save_best_only=True, save_weights_only=False
)
early = keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)

print("\n=== Stage 1: training (backbone frozen) ===")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weight,
    callbacks=[ckpt, early],
)


# In[9]:


# Stage 2 (fine-tuning top layers)
# =========================
print("\n=== Stage 2: fine-tuning (unfreeze sebagian) ===")
# ambil backbone dari dalam model agar robust
backbone = None
for l in model.layers:
    if "mobilenet" in l.name.lower():
        backbone = l
        break
if backbone is None:
    raise RuntimeError("Backbone MobileNetV2 tidak ditemukan dalam model.")

backbone.trainable = True
for l in backbone.layers[:-40]:
    l.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weight,
    callbacks=[ckpt, early],
)


# In[ ]:


# Evaluasi best model
best = keras.models.load_model(str(MODEL_PATH))
_, val_acc = best.evaluate(val_ds, verbose=0)
print(f"Best val acc: {val_acc:.4f}")


# In[ ]:


# RandomForest di atas embedding CNN
# =========================
print("
=== Train RandomForest on CNN embeddings ===")
emb_extractor = keras.Model(best.input, best.get_layer("gap").output)


def ds_to_emb(ds):
X, y = [], []
for xb, yb in ds:
X.append(emb_extractor.predict(xb, verbose=0))
y.append(yb.numpy())
return np.vstack(X), np.concatenate(y)


Xtr, ytr = ds_to_emb(train_ds)
Xva, yva = ds_to_emb(val_ds)


rf = RandomForestClassifier(n_estimators=400, n_jobs=-1,
class_weight="balanced_subsample", random_state=SEED)
rf.fit(Xtr, ytr)
print("RF val acc:", rf.score(Xva, yva))
joblib.dump(rf, RF_PATH)


# In[ ]:


# Disease info skeleton (bila belum ada)
# =========================
info_path = ARTIFACTS_DIR/"disease_info.json"
if not info_path.exists():
template = {c: {"nama": c.replace("_"," "), "deskripsi":"", "pengobatan":"", "perawatan":"", "pencegahan":""}
for c in class_names}
info_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
print("Created skeleton:", info_path)


print("
Training selesai. Tersimpan:")
print(" -", MODEL_PATH.resolve())
print(" -", RF_PATH.resolve())
print(" -", (ARTIFACTS_DIR/'class_indices.json').resolve())
print(" -", info_path.resolve())

