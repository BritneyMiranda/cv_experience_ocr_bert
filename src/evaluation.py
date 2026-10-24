"""Métricas, guardado de predicciones e intervalos bootstrap.

Cada modelo guarda sus predicciones en output/metrics/predictions/<modelo>.csv con
columnas cv_id, split (val | test | test_ref), y_true, y_pred. "test_ref" son los
mismos CVs de test pero con el texto de referencia en vez del OCR (Pregunta 2).

Uso:
    python -m src.evaluation --split val     # tabla de validación (durante el desarrollo)
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score

from src.utils import (DEFAULT_TEXT_COL, ID_COL, METRICS_DIR, PREDICTIONS_DIR, SEED,
                       TABLE_DIR, clip_years, load_json, save_json, to_band)


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), clip_years(y_pred)
    err = np.abs(y_pred - y_true)
    bands_true, bands_pred = to_band(y_true), to_band(y_pred)
    return {
        "MAE": err.mean(),
        "RMSE": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "Acc_pm1": float((err <= 1).mean()),
        "Acc_pm2": float((err <= 2).mean()),
        "Acc_pm3": float((err <= 3).mean()),
        "Band_acc": accuracy_score(bands_true, bands_pred),
        "Band_macroF1": f1_score(bands_true, bands_pred, average="macro"),
        "R2": r2_score(y_true, y_pred),
        "Mean_error": float((y_pred - y_true).mean()),
    }


def paired_bootstrap_mae_gain(y_true, pred_base, pred_new, n_boot: int = 2000,
                              seed: int = SEED) -> dict:
    """Reducción de MAE (base − nuevo) con IC 95 % por bootstrap pareado: en cada
    remuestreo se usan los mismos CVs para ambos modelos."""
    y_true = np.asarray(y_true, float)
    e_base = np.abs(clip_years(pred_base) - y_true)
    e_new = np.abs(clip_years(pred_new) - y_true)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    gains = e_base[idx].mean(axis=1) - e_new[idx].mean(axis=1)
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {"gain": float(e_base.mean() - e_new.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "significant": bool(lo > 0 or hi < 0), "n_boot": n_boot}


def model_name(base: str, text_col: str) -> str:
    """Nombre del archivo de predicciones; las variantes de texto (Pregunta 3) llevan sufijo."""
    return base if text_col == DEFAULT_TEXT_COL else f"{base}__{text_col}"


def run_and_save(name: str, predict_fn, frames: dict, text_col: str,
                 train_seconds: float, extra: dict | None = None) -> pd.DataFrame:
    """Predice val y test (y test con texto de referencia), guarda predicciones y
    tiempos, e imprime SOLO las métricas de validación."""
    parts, test_seconds = [], None
    jobs = [("val", "val", text_col), ("test", "test", text_col)]
    if text_col == DEFAULT_TEXT_COL:
        jobs.append(("test_ref", "test", "text_ref"))
    for label, split, col in jobs:
        df = frames[split]
        t0 = time.time()
        pred = clip_years(predict_fn(df[col].tolist()))
        if label == "test":
            test_seconds = time.time() - t0
        parts.append(pd.DataFrame({ID_COL: df[ID_COL], "split": label,
                                   "y_true": df["y"], "y_pred": pred}))
    preds = pd.concat(parts, ignore_index=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PREDICTIONS_DIR / f"{name}.csv", index=False)

    timing = {"model": name, "text_col": text_col,
              "train_seconds": round(train_seconds, 2),
              "inference_seconds_per_cv": test_seconds / len(frames["test"]),
              **(extra or {})}
    save_json(timing, METRICS_DIR / "timing" / f"{name}.json")

    val = preds[preds["split"] == "val"]
    m = compute_metrics(val["y_true"], val["y_pred"])
    print(f"[{name}] VALIDACIÓN  MAE {m['MAE']:.3f} | RMSE {m['RMSE']:.3f} | "
          f"±2 {m['Acc_pm2']:.1%} | R² {m['R2']:.3f} | entrenamiento {train_seconds:.1f} s")
    return preds


def comparison_table(split: str) -> pd.DataFrame:
    rows = []
    for path in sorted(PREDICTIONS_DIR.glob("*.csv")):
        preds = pd.read_csv(path)
        part = preds[preds["split"] == split]
        if part.empty:
            continue
        timing_path = METRICS_DIR / "timing" / f"{path.stem}.json"
        timing = load_json(timing_path) if timing_path.exists() else {}
        rows.append({"Model": path.stem, **compute_metrics(part["y_true"], part["y_pred"]),
                     "Training time (s)": timing.get("train_seconds"),
                     "Inference time per CV (s)": timing.get("inference_seconds_per_cv")})
    return pd.DataFrame(rows).sort_values("MAE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=["val", "test", "test_ref"])
    args = parser.parse_args()
    table = comparison_table(args.split)
    table.round(4).to_csv(TABLE_DIR / f"model_comparison_{args.split}.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
