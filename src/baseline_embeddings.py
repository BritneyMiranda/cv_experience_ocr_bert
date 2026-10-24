"""Baseline 2 — Sentence embeddings (encoder congelado) + Ridge.

Cada CV se codifica con all-MiniLM-L6-v2 (384 dimensiones, límite de 256 tokens). El
encoder NO se ajusta: solo se entrena una Ridge sobre los vectores.

CVs más largos que el límite: se parten en trozos de líneas completas de hasta 250
tokens (sin cortar una línea, así un cargo no se separa de sus fechas), se codifica cada
trozo y se promedian los vectores (ponderando por número de tokens).

Uso:
    python -m src.baseline_embeddings
    python -m src.baseline_embeddings --text-col text_raw
"""

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.evaluation import model_name, run_and_save
from src.utils import (BASELINES_DIR, DEFAULT_TEXT_COL, TABLE_DIR, TEXT_COLS, clip_years,
                       load_splits, set_seed)

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENS = 250
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


class ChunkedEncoder:
    def __init__(self, name: str = ENCODER):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)
        self.tokenizer = self.model.tokenizer
        self.n_chunked = 0

    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def chunks(self, text: str) -> list[str]:
        if self.n_tokens(text) <= CHUNK_TOKENS:
            return [text]
        self.n_chunked += 1
        out, current, size = [], [], 0
        for line in text.split("\n"):
            n = self.n_tokens(line)
            if current and size + n > CHUNK_TOKENS:
                out.append("\n".join(current))
                current, size = [], 0
            current.append(line)
            size += n
        out.append("\n".join(current))
        return out

    def encode(self, texts: list[str]) -> np.ndarray:
        pieces, owner, weights = [], [], []
        for i, text in enumerate(texts):
            for chunk in self.chunks(text):
                pieces.append(chunk)
                owner.append(i)
                weights.append(max(self.n_tokens(chunk), 1))
        vecs = self.model.encode(pieces, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False)
        out = np.zeros((len(texts), vecs.shape[1]))
        total = np.zeros(len(texts))
        for v, i, w in zip(vecs, owner, weights):
            out[i] += w * v
            total[i] += w
        return out / total[:, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-col", default=DEFAULT_TEXT_COL, choices=TEXT_COLS)
    args = parser.parse_args()
    set_seed()

    name = model_name("embeddings", args.text_col)
    frames = load_splits()
    encoder = ChunkedEncoder()

    t0 = time.time()
    X_tr = encoder.encode(frames["train"][args.text_col].tolist())
    encode_train_seconds = time.time() - t0
    X_va = encoder.encode(frames["val"][args.text_col].tolist())
    lengths = [encoder.n_tokens(t) for t in frames["train"][args.text_col]]
    print(f"Tokens por CV (train): mediana {np.median(lengths):.0f}, máx {max(lengths)}; "
          f"CVs partidos en trozos (train+val): {encoder.n_chunked}")

    grid = []
    for alpha in ALPHAS:
        ridge = Ridge(alpha=alpha).fit(X_tr, frames["train"]["y"])
        grid.append({"alpha": alpha, "val_MAE": mean_absolute_error(
            frames["val"]["y"], clip_years(ridge.predict(X_va)))})
    grid = pd.DataFrame(grid).sort_values("val_MAE")
    grid.to_csv(TABLE_DIR / f"grid_{name}.csv", index=False)
    alpha = float(grid.iloc[0]["alpha"])
    print(f"Mejor alpha en validación: {alpha}")

    t0 = time.time()
    ridge = Ridge(alpha=alpha).fit(X_tr, frames["train"]["y"])
    train_seconds = encode_train_seconds + (time.time() - t0)
    joblib.dump({"encoder": ENCODER, "ridge": ridge}, BASELINES_DIR / f"{name}.joblib")

    run_and_save(name, lambda texts: ridge.predict(encoder.encode(texts)), frames,
                 args.text_col, train_seconds,
                 extra={"encoder": ENCODER, "alpha": alpha,
                        "tokens_median": float(np.median(lengths)),
                        "tokens_max": int(max(lengths)),
                        "cvs_chunked": encoder.n_chunked})


if __name__ == "__main__":
    main()
