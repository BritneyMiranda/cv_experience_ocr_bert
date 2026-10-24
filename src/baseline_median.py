"""Baseline 0 — Predice siempre la mediana de años del set de entrenamiento.

No lee el texto: es el piso que cualquier otro modelo debe superar.

Uso:
    python -m src.baseline_median
"""

import time

import numpy as np

from src.evaluation import run_and_save
from src.utils import BASELINES_DIR, DEFAULT_TEXT_COL, load_splits, save_json


def main() -> None:
    frames = load_splits()
    t0 = time.time()
    median = float(np.median(frames["train"]["y"]))
    train_seconds = time.time() - t0
    save_json({"median": median}, BASELINES_DIR / "median.json")
    print(f"Mediana de entrenamiento: {median:.2f} años")
    run_and_save("median", lambda texts: np.full(len(texts), median), frames,
                 DEFAULT_TEXT_COL, train_seconds)


if __name__ == "__main__":
    main()
