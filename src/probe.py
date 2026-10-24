"""Pregunta 1 — Prueba de fechas: ¿el modelo lee las fechas o solo el vocabulario?

Sobre 20 CVs de test con al menos dos trabajos se aplican tres cambios y se compara
cómo se mueve cada predicción con lo que haría un reclutador:
  - shift_+5: todas las fechas (trabajos y estudios) +5 años. Las duraciones no
    cambian, así que un reclutador no cambiaría su estimación (Δ esperado = 0).
  - shift_-5: lo mismo hacia el pasado (Δ esperado = 0).
  - delete_job: se borra el trabajo más antiguo (su línea y sus viñetas). Un reclutador
    restaría su duración: Δ esperado = suma de periodos después − antes.

Uso:
    python -m src.probe
"""

import re

import numpy as np
import pandas as pd

from src.baseline_rules import merged_months, periods
from src.predict import LABELS, load_predictors
from src.utils import ID_COL, SEED, TABLE_DIR, clip_years, load_splits

N_CVS = 20
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def shift_years(text: str, delta: int) -> str:
    return YEAR_RE.sub(lambda m: str(int(m.group(1)) + delta), text)


def delete_oldest_job(text: str) -> str:
    """Borra la línea con el periodo más antiguo y las viñetas que la siguen."""
    lines = text.split("\n")
    job_lines = [i for i, l in enumerate(lines) if periods(l)]
    oldest = min(job_lines, key=lambda i: periods(lines[i])[0][0])
    end = oldest + 1
    while end < len(lines) and lines[end].startswith("- "):
        end += 1
    return "\n".join(lines[:oldest] + lines[end:])


def years_from_dates(text: str) -> float:
    """Lo que calcularía un reclutador, con el mismo tope de 20 años que el target."""
    return float(clip_years(merged_months(periods(text)) / 12))


def main() -> None:
    test = load_splits()["test"]
    test = test[test["text_clean"].map(lambda t: len(periods(t)) >= 2)]
    sample = test.sample(N_CVS, random_state=SEED)
    original = sample["text_clean"].tolist()
    variants = {
        "shift_+5": [shift_years(t, 5) for t in original],
        "shift_-5": [shift_years(t, -5) for t in original],
        "delete_job": [delete_oldest_job(t) for t in original],
    }
    expected = {
        "shift_+5": np.zeros(N_CVS),
        "shift_-5": np.zeros(N_CVS),
        "delete_job": np.array([years_from_dates(v) - years_from_dates(o)
                                for o, v in zip(original, variants["delete_job"])]),
    }

    rows = []
    for name, predict in load_predictors().items():
        base = predict(original)
        for change, texts in variants.items():
            delta = predict(texts) - base
            for cv_id, d, e in zip(sample[ID_COL], delta, expected[change]):
                rows.append({"model": LABELS[name], "change": change, ID_COL: cv_id,
                             "delta_pred": d, "delta_expected": e})
    detail = pd.DataFrame(rows)
    detail.to_csv(TABLE_DIR / "q1_probe_detail.csv", index=False)

    detail["abs_dev"] = (detail["delta_pred"] - detail["delta_expected"]).abs()
    summary = detail.groupby(["change", "model"]).agg(
        mean_delta_pred=("delta_pred", "mean"), mean_delta_expected=("delta_expected", "mean"),
        mean_abs_deviation=("abs_dev", "mean")).reset_index()
    summary.to_csv(TABLE_DIR / "q1_probe_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    example = sample.iloc[0]
    print(f"\nEjemplo ({example[ID_COL]}), borrar el trabajo más antiguo:\n"
          f"{variants['delete_job'][0]}")


if __name__ == "__main__":
    main()
