"""Carga los modelos entrenados y predice años de experiencia sobre texto nuevo.

Lo usan la prueba de la Pregunta 1 (src/probe.py) y la app de Streamlit. Cada
predictor recibe una lista de textos ya limpios y devuelve años recortados a [0, 20].
"""

import numpy as np

from src.utils import BASELINES_DIR, BERT_DIR, clip_years, load_json

LABELS = {"median": "Median", "tfidf": "TF-IDF + Ridge",
          "embeddings": "Embeddings + Ridge", "bert": "Fine-tuned BERT",
          "rules": "Rule-based (bonus)"}


def _median():
    median = load_json(BASELINES_DIR / "median.json")["median"]
    return lambda texts: np.full(len(texts), median)


def _tfidf():
    import joblib
    pipe = joblib.load(BASELINES_DIR / "tfidf.joblib")
    return pipe.predict


def _embeddings():
    import joblib
    from src.baseline_embeddings import ChunkedEncoder
    saved = joblib.load(BASELINES_DIR / "embeddings.joblib")
    encoder, ridge = ChunkedEncoder(saved["encoder"]), saved["ridge"]
    return lambda texts: ridge.predict(encoder.encode(texts))


def _bert(model_dir=None, max_length: int = 256):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = str(model_dir or BERT_DIR / "bert")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()

    @torch.no_grad()
    def predict(texts):
        out = []
        for i in range(0, len(texts), 64):
            enc = tokenizer(list(texts[i:i + 64]), truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
            out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
        return np.concatenate(out) * 20
    return predict


def _rules():
    from src.baseline_rules import predict
    return predict


LOADERS = {"median": _median, "tfidf": _tfidf, "embeddings": _embeddings,
           "bert": _bert, "rules": _rules}


def load_predictors(names=tuple(LOADERS)) -> dict:
    """{nombre: función(textos) -> años} para los modelos cuyos archivos existan."""
    predictors = {}
    for name in names:
        try:
            fn = LOADERS[name]()
        except (OSError, ImportError) as e:  # archivos o librerías no disponibles
            print(f"[{name}] no disponible: {e}")
            continue
        predictors[name] = (lambda f: lambda texts: clip_years(f(list(texts))))(fn)
    return predictors
