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

from src.utils import (DEFAULT_TEXT_COL, ID_COL, METRICS_DIR, PREDICTIONS_DIR,
                       PROCESSED_CSV, SEED, TABLE_DIR, clip_years, load_json, save_json,
                       to_band)


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


# --- Evaluación final en test -------------------------------------------------------

MAIN_MODELS = {"median": "Median", "tfidf": "TF-IDF + Ridge",
               "embeddings": "Embeddings + Ridge", "bert": "Fine-tuned BERT",
               "rules": "Rule-based (bonus)"}
BASELINES = ["median", "tfidf", "embeddings", "rules"]
BAND_ORDER = ["0-2", "3-5", "6-10", "11-20"]


def load_test(name: str, split: str = "test") -> pd.DataFrame:
    """Predicciones de test de un modelo unidas con ocupación, sexo y raza."""
    preds = pd.read_csv(PREDICTIONS_DIR / f"{name}.csv", dtype={ID_COL: str})
    preds = preds[preds["split"] == split]
    cvs = pd.read_csv(PROCESSED_CSV, dtype={ID_COL: str},
                      usecols=[ID_COL, "occupation", "sex", "race"])
    df = preds.merge(cvs, on=ID_COL, how="left")
    df["band"] = to_band(df["y_true"])
    df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    df["err"] = df["y_pred"] - df["y_true"]
    return df


def model_comparison() -> pd.DataFrame:
    rows = []
    for name, label in MAIN_MODELS.items():
        df = load_test(name)
        m = compute_metrics(df["y_true"], df["y_pred"])
        timing = load_json(METRICS_DIR / "timing" / f"{name}.json")
        rows.append({
            "Model": label, "MAE (years)": m["MAE"], "RMSE (years)": m["RMSE"],
            "Acc ±1": m["Acc_pm1"], "Acc ±2": m["Acc_pm2"], "Acc ±3": m["Acc_pm3"],
            "Band accuracy": m["Band_acc"], "Band macro-F1": m["Band_macroF1"], "R²": m["R2"],
            "Training time": f"{timing['train_seconds']:.1f} s",
            "Inference time per CV": f"{timing['inference_seconds_per_cv'] * 1000:.2f} ms",
        })
    return pd.DataFrame(rows)


def bert_gains() -> pd.DataFrame:
    bert = load_test("bert").set_index(ID_COL)
    rows = []
    for base in BASELINES:
        other = load_test(base).set_index(ID_COL).loc[bert.index]
        g = paired_bootstrap_mae_gain(bert["y_true"], other["y_pred"], bert["y_pred"])
        rows.append({"baseline": MAIN_MODELS[base], "MAE baseline": other["abs_err"].mean(),
                     "MAE BERT": bert["abs_err"].mean(), "MAE gain (baseline - BERT)": g["gain"],
                     "CI 95% low": g["ci_low"], "CI 95% high": g["ci_high"],
                     "significant": g["significant"], "n_boot": g["n_boot"]})
    return pd.DataFrame(rows)


def grouped(col: str, models=MAIN_MODELS) -> pd.DataFrame:
    """MAE y error medio (predicho − real) de cada modelo por grupo."""
    rows = []
    for name, label in models.items():
        df = load_test(name)
        for group, g in df.groupby(col):
            rows.append({"model": label, col: group, "n": len(g),
                         "MAE": g["abs_err"].mean(), "mean_error": g["err"].mean()})
    return pd.DataFrame(rows)


def confusion(name: str) -> pd.DataFrame:
    df = load_test(name)
    df["band_pred"] = to_band(df["y_pred"])
    return pd.crosstab(df["band"], df["band_pred"]).reindex(
        index=BAND_ORDER, columns=BAND_ORDER, fill_value=0)


def ocr_vs_reference() -> pd.DataFrame:
    """Pregunta 2: el mismo modelo sobre el OCR de test y sobre el texto de referencia."""
    rows = []
    for name, label in MAIN_MODELS.items():
        ocr, ref = load_test(name, "test"), load_test(name, "test_ref")
        rows.append({"model": label, "MAE OCR text": ocr["abs_err"].mean(),
                     "MAE reference text": ref["abs_err"].mean(),
                     "gap (OCR - reference)": ocr["abs_err"].mean() - ref["abs_err"].mean()})
    return pd.DataFrame(rows)


def sensitivity() -> pd.DataFrame:
    """Pregunta 3: versión del texto (y dígitos en TF-IDF, pérdida en BERT) en test."""
    experiments = [
        ("OCR cleaning", "raw OCR", "{m}__text_raw"),
        ("OCR cleaning", "cleaned OCR (baseline setup)", "{m}"),
        ("OCR cleaning", "cleaned, low-confidence lines removed", "{m}__text_clean_conf"),
    ]
    rows = []
    for exp, setup, pattern in experiments:
        for m in ["tfidf", "embeddings", "bert", "rules"]:
            rows.append({"experiment": exp, "setup": setup, "model": MAIN_MODELS[m],
                         "MAE": load_test(pattern.format(m=m))["abs_err"].mean()})
    for exp, setup, name, m in [
        ("Numbers in TF-IDF", "keep digits (baseline setup)", "tfidf", "tfidf"),
        ("Numbers in TF-IDF", "remove digits", "tfidf_nodigits", "tfidf"),
        ("BERT loss", "MSE (baseline setup)", "bert", "bert"),
        ("BERT loss", "Huber", "bert_huber", "bert"),
    ]:
        rows.append({"experiment": exp, "setup": setup, "model": MAIN_MODELS[m],
                     "MAE": load_test(name)["abs_err"].mean()})
    return pd.DataFrame(rows)


def fairness(col: str, n_boot: int = 2000) -> pd.DataFrame:
    """Pregunta 4: MAE y error medio de BERT por grupo, con IC 95 % del error medio."""
    df = load_test("bert")
    rng = np.random.default_rng(SEED)
    rows = []
    for group, g in df.groupby(col):
        err = g["err"].to_numpy()
        boots = err[rng.integers(0, len(err), size=(n_boot, len(err)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({col: group, "n": len(g), "MAE": g["abs_err"].mean(),
                     "mean_error (pred - true)": err.mean(), "CI 95% low": lo,
                     "CI 95% high": hi, "mean_true_years": g["y_true"].mean()})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def top_errors(n: int = 10) -> pd.DataFrame:
    """Los n mayores errores de BERT con señales para clasificar la causa."""
    from src.baseline_rules import merged_months, periods
    bert = load_test("bert")
    rules = load_test("rules").set_index(ID_COL)["y_pred"]
    cvs = pd.read_csv(PROCESSED_CSV, dtype={ID_COL: str}).set_index(ID_COL)
    rows = []
    for _, r in bert.nlargest(n, "abs_err").iterrows():
        text, ref = cvs.loc[r[ID_COL], "text_clean"], cvs.loc[r[ID_COL], "text_ref"]
        ps = sorted(periods(text))
        span = (max(e for _, e in ps) - min(s for s, _ in ps)) if ps else 0
        rows.append({
            ID_COL: r[ID_COL], "occupation": r["occupation"], "y_true": r["y_true"],
            "bert": r["y_pred"], "error": r["err"], "rules_on_ocr": rules[r[ID_COL]],
            "n_jobs": len(ps),
            "gap_years": (span - merged_months(ps)) / 12,
            "overlap": sum(e for _, e in ps) - sum(s for s, _ in ps) > merged_months(ps),
            "ocr_dates_match_reference": periods(text) == periods(ref),
            "text": text,
        })
    return pd.DataFrame(rows)


def final_evaluation() -> None:
    table = model_comparison()
    table.to_csv(TABLE_DIR / "model_comparison.csv", index=False)
    print("TEST\n", table.round(3).to_string(index=False))

    gains = bert_gains()
    gains.to_csv(TABLE_DIR / "bert_gain_bootstrap.csv", index=False)
    print("\nGanancia de BERT (bootstrap pareado)\n", gains.round(3).to_string(index=False))

    grouped("band").to_csv(TABLE_DIR / "mae_by_band.csv", index=False)
    grouped("occupation").to_csv(TABLE_DIR / "mae_by_occupation.csv", index=False)
    for name in MAIN_MODELS:
        confusion(name).to_csv(TABLE_DIR / f"band_confusion_{name}.csv")

    q2 = ocr_vs_reference()
    q2.to_csv(TABLE_DIR / "q2_ocr_vs_reference.csv", index=False)
    print("\nPregunta 2\n", q2.round(3).to_string(index=False))

    q3 = sensitivity()
    q3.to_csv(TABLE_DIR / "q3_sensitivity.csv", index=False)
    print("\nPregunta 3\n", q3.round(3).to_string(index=False))

    for col in ["sex", "race"]:
        f = fairness(col)
        f.to_csv(TABLE_DIR / f"q4_fairness_{col}.csv", index=False)
        print(f"\nPregunta 4 ({col})\n", f.round(3).to_string(index=False))

    top_errors().to_csv(TABLE_DIR / "bert_top10_errors.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=["val", "test", "test_ref"])
    parser.add_argument("--final", action="store_true",
                        help="evaluación final en test (tablas de las secciones 9 y 10)")
    args = parser.parse_args()
    if args.final:
        final_evaluation()
        return
    table = comparison_table(args.split)
    table.round(4).to_csv(TABLE_DIR / f"model_comparison_{args.split}.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
