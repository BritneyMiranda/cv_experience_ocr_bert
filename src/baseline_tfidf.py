"""Baseline 1 — TF-IDF + Ridge.

Bolsa de n-gramas de palabras ponderada con TF-IDF y regresión Ridge. Los números se
conservan como tokens por defecto: "2018.12 - 2023.01" se tokeniza como 2018, 12, 2023,
01, así el modelo puede asociar años de inicio tempranos con más experiencia, aunque no
puede restar fechas. Con --no-digits se eliminan (Pregunta 3).

Hiperparámetros elegidos en validación: rango de n-gramas y alpha de Ridge.

Uso:
    python -m src.baseline_tfidf
    python -m src.baseline_tfidf --no-digits
    python -m src.baseline_tfidf --text-col text_raw
"""

import argparse
import itertools
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from src.evaluation import model_name, run_and_save
from src.utils import (BASELINES_DIR, DEFAULT_TEXT_COL, TABLE_DIR, TEXT_COLS, clip_years,
                       load_splits)

ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
NGRAMS = [(1, 1), (1, 2), (1, 3)]


def build(ngram: tuple, alpha: float, keep_digits: bool) -> Pipeline:
    token_pattern = r"(?u)\b\w+\b" if keep_digits else r"(?u)\b[^\W\d_]{2,}\b"
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram, token_pattern=token_pattern,
                                  min_df=2, sublinear_tf=True)),
        ("ridge", Ridge(alpha=alpha)),
    ])


def top_features(pipe: Pipeline, n: int = 20) -> pd.DataFrame:
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = pipe.named_steps["ridge"].coef_
    order = np.argsort(coef)
    rows = [{"feature": names[i], "coef": coef[i], "direction": "menos años"} for i in order[:n]]
    rows += [{"feature": names[i], "coef": coef[i], "direction": "más años"} for i in order[::-1][:n]]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-col", default=DEFAULT_TEXT_COL, choices=TEXT_COLS)
    parser.add_argument("--no-digits", action="store_true")
    args = parser.parse_args()

    base = "tfidf_nodigits" if args.no_digits else "tfidf"
    name = model_name(base, args.text_col)
    frames = load_splits()
    X_tr, y_tr = frames["train"][args.text_col], frames["train"]["y"]
    X_va, y_va = frames["val"][args.text_col], frames["val"]["y"]

    grid = []
    for ngram, alpha in itertools.product(NGRAMS, ALPHAS):
        pipe = build(ngram, alpha, not args.no_digits).fit(X_tr, y_tr)
        mae = mean_absolute_error(y_va, clip_years(pipe.predict(X_va)))
        grid.append({"ngram_max": ngram[1], "alpha": alpha, "val_MAE": mae})
    grid = pd.DataFrame(grid).sort_values("val_MAE")
    grid.to_csv(TABLE_DIR / f"grid_{name}.csv", index=False)
    best = grid.iloc[0]
    ngram = (1, int(best["ngram_max"]))
    print(f"Mejor en validación: n-gramas {ngram}, alpha {best['alpha']}")

    t0 = time.time()
    pipe = build(ngram, best["alpha"], not args.no_digits).fit(X_tr, y_tr)
    train_seconds = time.time() - t0
    joblib.dump(pipe, BASELINES_DIR / f"{name}.joblib")
    top_features(pipe).to_csv(TABLE_DIR / f"top_features_{name}.csv", index=False)

    run_and_save(name, pipe.predict, frames, args.text_col, train_seconds,
                 extra={"ngram": list(ngram), "alpha": float(best["alpha"]),
                        "n_features": len(pipe.named_steps["tfidf"].vocabulary_)})


if __name__ == "__main__":
    main()
