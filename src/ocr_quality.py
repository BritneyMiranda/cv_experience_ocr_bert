"""Fase 3 — Calidad del OCR frente al texto de referencia.

Métricas por CV:
  - CER y WER con jiwer, en dos variantes: "raw" (solo se colapsan espacios) y
    "norm" (misma normalización para ambos textos: minúsculas, NFKC, todos los
    guiones como "-", sin viñetas ni "|"). La normalizada mide errores de lectura;
    la cruda incluye además diferencias de puntuación.
  - Años leídos correctamente: de los años (19xx/20xx) del texto de referencia, qué
    proporción aparece en el OCR. Se calcula sobre el OCR crudo y tras reparar los
    rangos de fechas sin guion (ver preprocessing.repair_date_ranges).
  - Rangos de fechas correctos: proporción de periodos AAAA.MM–AAAA.MM de la referencia
    que el OCR reproduce exactamente.

Uso:
    python -m src.ocr_quality
"""

import re
import unicodedata
from collections import Counter

import jiwer
import numpy as np
import pandas as pd

from src.preprocessing import DATE_RANGE_RE, YEAR_RE, repair_date_ranges
from src.utils import (ID_COL, METADATA_CSV, METRICS_DIR, OCR_DIR, REFERENCE_TEXT_CSV,
                       TABLE_DIR, load_json, save_json)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_eval(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[‐‑‒–—―−]", "-", text)
    text = re.sub(r"[•·▪●◦|]", " ", text)
    return normalize_ws(text)


def share_found(reference: list, hypothesis: list) -> float:
    """Proporción de elementos de la referencia presentes en la hipótesis (multiconjunto)."""
    if not reference:
        return np.nan
    found = Counter(reference) & Counter(hypothesis)
    return sum(found.values()) / len(reference)


def ranges(text: str) -> list[tuple[str, str]]:
    return DATE_RANGE_RE.findall(unicodedata.normalize("NFKC", text))


def evaluate_cv(ref: str, ocr: str) -> dict:
    ocr_repaired, n_missing_dash = repair_date_ranges(re.sub(r"[‐‑‒–—―−]", "-", ocr))
    ref_years = YEAR_RE.findall(ref)
    return {
        "cer_raw": jiwer.cer(normalize_ws(ref), normalize_ws(ocr)),
        "wer_raw": jiwer.wer(normalize_ws(ref), normalize_ws(ocr)),
        "cer": jiwer.cer(normalize_eval(ref), normalize_eval(ocr)),
        "wer": jiwer.wer(normalize_eval(ref), normalize_eval(ocr)),
        "n_ref_years": len(ref_years),
        "years_ok_raw": share_found(ref_years, YEAR_RE.findall(ocr)),
        "years_ok_repaired": share_found(ref_years, YEAR_RE.findall(ocr_repaired)),
        "n_ref_ranges": len(ranges(ref)),
        "ranges_ok_repaired": share_found(ranges(ref), ranges(ocr_repaired)),
        "n_missing_dash": n_missing_dash,
    }


def top_word_errors(pairs: list[tuple[str, str]], n: int = 25) -> pd.DataFrame:
    """Sustituciones de palabras más frecuentes (texto normalizado)."""
    subs = Counter()
    for ref, hyp in pairs:
        out = jiwer.process_words(normalize_eval(ref), normalize_eval(hyp))
        r_words, h_words = out.references[0], out.hypotheses[0]
        for chunk in out.alignments[0]:
            if chunk.type == "substitute":
                for i, j in zip(range(chunk.ref_start_idx, chunk.ref_end_idx),
                                range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                    subs[(r_words[i], h_words[j])] += 1
    return pd.DataFrame([{"referencia": r, "ocr": h, "veces": c}
                         for (r, h), c in subs.most_common(n)])


def main() -> None:
    meta = pd.read_csv(METADATA_CSV, dtype={ID_COL: str})[[ID_COL, "occupation"]]
    ref = pd.read_csv(REFERENCE_TEXT_CSV, dtype=str).set_index(ID_COL)["text"]

    rows, pairs = [], []
    for cv_id, occupation in zip(meta[ID_COL], meta["occupation"]):
        path = OCR_DIR / f"{cv_id}.json"
        if not path.exists() or cv_id not in ref.index:
            continue
        record = load_json(path)
        pairs.append((ref[cv_id], record["text"]))
        rows.append({ID_COL: cv_id, "occupation": occupation,
                     "mean_conf": record["mean_score"],
                     "min_conf": min((l["score"] for l in record["lines"]), default=np.nan),
                     **evaluate_cv(ref[cv_id], record["text"])})
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "ocr_quality_per_cv.csv", index=False)

    cols = ["cer_raw", "wer_raw", "cer", "wer", "years_ok_raw", "years_ok_repaired",
            "ranges_ok_repaired", "mean_conf"]
    by_occ = df.groupby("occupation")[cols].mean()
    by_occ.loc["ALL"] = df[cols].mean()
    by_occ.round(4).to_csv(TABLE_DIR / "ocr_quality_by_occupation.csv")
    print(by_occ.round(4).to_string())

    errors = top_word_errors(pairs)
    errors.to_csv(TABLE_DIR / "ocr_top_word_errors.csv", index=False)
    print("\nSustituciones más frecuentes:\n", errors.head(15).to_string(index=False))

    # Años correctos ponderados por número de años (no promedio de proporciones)
    def pooled(col: str, n_col: str) -> float:
        return float((df[col] * df[n_col]).sum() / df[n_col].sum())

    summary = {
        "n_cvs": len(df),
        "cer_mean": df["cer"].mean(), "cer_median": df["cer"].median(),
        "wer_mean": df["wer"].mean(), "wer_median": df["wer"].median(),
        "cer_raw_mean": df["cer_raw"].mean(), "wer_raw_mean": df["wer_raw"].mean(),
        "cer_p95": df["cer"].quantile(0.95),
        "years_ok_raw": pooled("years_ok_raw", "n_ref_years"),
        "years_ok_repaired": pooled("years_ok_repaired", "n_ref_years"),
        "ranges_ok_repaired": pooled("ranges_ok_repaired", "n_ref_ranges"),
        "cvs_with_missing_dash": int((df["n_missing_dash"] > 0).sum()),
        "ranges_missing_dash": int(df["n_missing_dash"].sum()),
        "cvs_all_years_ok": int((df["years_ok_repaired"] == 1).sum()),
        "corr_conf_vs_cer": float(df[["mean_conf", "cer"]].corr().iloc[0, 1]),
    }
    summary = {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}
    save_json(summary, METRICS_DIR / "ocr_quality_summary.json")
    print("\n", summary)


if __name__ == "__main__":
    main()
