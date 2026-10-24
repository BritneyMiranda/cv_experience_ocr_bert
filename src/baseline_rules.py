"""Bonus — Modelo de reglas: suma la duración de los trabajos a partir de las fechas.

1. Busca con regex los periodos AAAA.MM – AAAA.MM (también sin guion, como los deja a
   veces el OCR, y con "Present"/"Current" como fin, que se toma como la fecha de
   referencia).
2. Convierte cada periodo a un intervalo en meses y **une los solapamientos**, para no
   contar dos veces meses con dos trabajos a la vez.
3. Años = meses cubiertos / 12. No se entrena nada.

Si un mes es ilegible (p. ej. "2023.0" porque el OCR cortó el último dígito), se usa el
mes 6 (mitad de año). Es una prueba honesta de si BERT hace algo más que aritmética.

Uso:
    python -m src.baseline_rules
    python -m src.baseline_rules --text-col text_raw
"""

import argparse
import re

import numpy as np

from src.evaluation import model_name, run_and_save
from src.utils import DEFAULT_TEXT_COL, TEXT_COLS, load_splits

REFERENCE_DATE = (2023, 1)  # fecha en la que terminan los trabajos actuales en PopResume
PERIOD_RE = re.compile(
    r"((?:19|20)\d{2})\.(\d{1,2})"            # inicio AAAA.MM
    r"\s*[-–—~]?\s*"                           # guion (a veces omitido por el OCR)
    r"(?:((?:19|20)\d{2})(?:\.(\d{1,2}))?"     # fin AAAA.MM (mes puede faltar)
    r"|(present|current|now))", re.IGNORECASE)


def to_month(year: str, month: str | None) -> int:
    m = int(month) if month and 1 <= int(month) <= 12 else 6
    return int(year) * 12 + (m - 1)


def periods(text: str) -> list[tuple[int, int]]:
    out = []
    for y1, m1, y2, m2, present in PERIOD_RE.findall(text):
        start = to_month(y1, m1)
        end = to_month(*map(str, REFERENCE_DATE)) if present else to_month(y2, m2)
        if end > start:
            out.append((start, end))
    return out


def merged_months(intervals: list[tuple[int, int]]) -> int:
    """Meses cubiertos por la unión de intervalos [inicio, fin)."""
    total, cur_start, cur_end = 0, None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def predict(texts: list[str]) -> np.ndarray:
    return np.array([merged_months(periods(t)) / 12 for t in texts])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text-col", default=DEFAULT_TEXT_COL, choices=TEXT_COLS)
    args = parser.parse_args()

    frames = load_splits()
    no_dates = sum(len(periods(t)) == 0 for t in frames["val"][args.text_col])
    print(f"CVs de validación sin ningún periodo detectado: {no_dates}")
    run_and_save(model_name("rules", args.text_col), predict, frames, args.text_col, 0.0)


if __name__ == "__main__":
    main()
