"""Utilidades compartidas: semillas, rutas, constantes y helpers."""

import json
import os
import random
from pathlib import Path

import numpy as np

# --- Rutas -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
# En Colab, CV_STORAGE_DIR apunta a Google Drive para que los datos sobrevivan
# a los reinicios de sesión. En local se usa el propio repositorio.
STORAGE = Path(os.environ.get("CV_STORAGE_DIR", ROOT))

DATA_DIR = STORAGE / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_IMG_DIR = RAW_DIR / "images"
OCR_DIR = DATA_DIR / "ocr"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = STORAGE / "models"
BASELINES_DIR = MODELS_DIR / "baselines"
BERT_DIR = MODELS_DIR / "bert"

OUTPUT_DIR = STORAGE / "output"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
METRICS_DIR = OUTPUT_DIR / "metrics"

# Archivos centrales del pipeline
METADATA_CSV = RAW_DIR / "metadata.csv"
SAMPLED_IDS_CSV = RAW_DIR / "sampled_ids.csv"
REFERENCE_TEXT_CSV = RAW_DIR / "reference_text.csv"

# --- Constantes del proyecto -----------------------------------------------
SEED = 42
HF_DATASET = "sumin-yu/PopResume"
N_PER_OCCUPATION = 800

TARGET = "exp_year"
EXP_CAP = 20  # años; se revisa en download_data.py (regla del 15 %)

# Bandas de experiencia para band accuracy / macro-F1 (límites inclusivos)
BANDS = [(0, 2, "0-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, EXP_CAP, f"11-{EXP_CAP}")]

# Columnas que NUNCA pueden ser features (sex y race solo para fairness)
FORBIDDEN_FEATURES = ["age", "sex", "race", "name", "state", "region"]


def ensure_dirs() -> None:
    for d in [RAW_IMG_DIR, OCR_DIR, PROCESSED_DIR, SPLITS_DIR, BASELINES_DIR,
              BERT_DIR, FIG_DIR, TABLE_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clip_years(y, cap: int = EXP_CAP):
    """Recorta años de experiencia (reales o predichos) al rango [0, cap]."""
    return np.clip(np.asarray(y, dtype=float), 0, cap)


def to_band(y) -> np.ndarray:
    """Convierte años a etiquetas de banda. Los valores se redondean antes."""
    y = np.rint(clip_years(y))
    labels = np.empty(len(y), dtype=object)
    for lo, hi, name in BANDS:
        labels[(y >= lo) & (y <= hi)] = name
    return labels


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
