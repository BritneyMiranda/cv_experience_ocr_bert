"""Fase 2 — Extracción de texto con PaddleOCR, caché en JSON y orden de lectura.

Por cada CV se guarda un JSON en data/ocr/<cv_id>.json con:
  - cada línea detectada: texto, confianza y caja [x1, y1, x2, y2]
  - el texto reconstruido en orden de lectura (ver reading_order)
  - tiempo de OCR y número de columnas detectadas
Los CVs que ya tienen JSON se saltan, así el OCR se ejecuta una sola vez.

Uso:
    python -m src.ocr_extraction                         # todos los CVs muestreados
    python -m src.ocr_extraction --limit 40 --out-dir data/ocr_bench/en_mobile \
        --rec-model en_PP-OCRv5_mobile_rec               # benchmark de modelos
    python -m src.ocr_extraction --image cv.png --json-out cv.json   # una imagen (app)
"""

import argparse
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (ID_COL, METRICS_DIR, OCR_DIR, RAW_IMG_DIR, SAMPLED_IDS_CSV,
                       SEED, load_json, save_json)

DET_MODEL = "PP-OCRv5_server_det"
REC_MODEL = "en_PP-OCRv5_mobile_rec"


# --- Orden de lectura --------------------------------------------------------

MONTHS_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
                       r"|\b(present|current|now|to)\b", re.IGNORECASE)


def is_date_only(text: str) -> bool:
    """True si la línea es solo una fecha o rango ("2014 – 2019", "Jan 2020 - Present")."""
    rest = MONTHS_RE.sub("", text)
    return any(c.isdigit() for c in text) and not any(c.isalpha() for c in rest)


def find_gutter(lines: list[dict], page_width: int) -> float | None:
    """Busca el "pasillo" vertical vacío que separa dos columnas.

    Se ignoran las líneas anchas (> 50 % de la página: títulos a todo lo ancho) y se
    buscan franjas del eje x que casi ninguna otra línea cubre (se toleran unas pocas,
    como un nombre centrado), en la zona central (20–80 % del ancho) y de al menos 2 %
    del ancho. Una franja es un
    pasillo válido si a cada lado queda al menos 15 % de las líneas y cada lado es
    sobre todo texto: las fechas alineadas a la derecha de un cargo dejan una franja
    vacía parecida, pero una "columna" hecha solo de fechas no es una columna.
    Entre los pasillos válidos se elige el que reparte las líneas más equilibradamente.
    """
    narrow = [l for l in lines if (l["box"][2] - l["box"][0]) < 0.5 * page_width]
    if len(narrow) < 10:
        return None
    coverage = np.zeros(page_width + 1, dtype=int)
    for l in narrow:
        coverage[max(l["box"][0], 0):min(l["box"][2], page_width) + 1] += 1
    # Unas pocas líneas pueden cruzar el pasillo (p. ej. el nombre centrado arriba)
    covered = coverage > max(1, int(0.05 * len(narrow)))

    lo, hi = int(0.2 * page_width), int(0.8 * page_width)
    gaps, start = [], None
    for x in range(lo, hi + 1):
        if not covered[x] and start is None:
            start = x
        if (covered[x] or x == hi) and start is not None:
            if x - start >= 0.02 * page_width:
                gaps.append((start + x) / 2)
            start = None

    best, best_balance = None, 0
    for gutter in gaps:
        left = [l for l in narrow if l["box"][2] <= gutter]
        right = [l for l in narrow if l["box"][0] >= gutter]
        balance = min(len(left), len(right))
        if balance < 0.15 * len(narrow):
            continue
        if any(np.mean([is_date_only(l["text"]) for l in side]) >= 0.5 for side in (left, right)):
            continue
        if balance > best_balance:
            best, best_balance = gutter, balance
    return best


def group_rows(lines: list[dict]) -> list[str]:
    """Agrupa en una fila las cajas a la misma altura (p. ej. cargo ... 2014 – 2019)
    y deja una línea en blanco cuando el salto vertical indica un bloque nuevo."""
    if not lines:
        return []
    heights = [l["box"][3] - l["box"][1] for l in lines]
    tol = 0.5 * float(np.median(heights))
    lines = sorted(lines, key=lambda l: (l["box"][1] + l["box"][3]) / 2)

    rows, current = [], [lines[0]]
    for l in lines[1:]:
        yc = (l["box"][1] + l["box"][3]) / 2
        yc_row = np.mean([(c["box"][1] + c["box"][3]) / 2 for c in current])
        if abs(yc - yc_row) <= tol:
            current.append(l)
        else:
            rows.append(current)
            current = [l]
    rows.append(current)

    out, prev_bottom = [], None
    for row in rows:
        top = min(l["box"][1] for l in row)
        if prev_bottom is not None and top - prev_bottom > 3 * tol:
            out.append("")
        out.append(" ".join(l["text"] for l in sorted(row, key=lambda l: l["box"][0])))
        prev_bottom = max(l["box"][3] for l in row)
    return out


def reading_order(lines: list[dict], page_width: int) -> tuple[str, int]:
    """Texto en orden de lectura y número de columnas detectadas (1 o 2).

    Con dos columnas, las líneas que cruzan el pasillo (encabezados a todo lo ancho)
    actúan como separadores: entre dos de ellas se lee primero la columna izquierda
    completa y después la derecha. Así un cargo no se mezcla con la otra columna.
    """
    gutter = find_gutter(lines, page_width)
    if gutter is None:
        return "\n".join(group_rows(lines)), 1

    out, left, right = [], [], []
    for l in sorted(lines, key=lambda l: l["box"][1]):
        if l["box"][2] <= gutter:
            left.append(l)
        elif l["box"][0] >= gutter:
            right.append(l)
        else:  # línea a todo lo ancho: cierra el bloque anterior
            out += group_rows(left) + group_rows(right) + [l["text"]]
            left, right = [], []
    out += group_rows(left) + [""] + group_rows(right)
    return "\n".join(out).strip(), 2


# --- OCR -----------------------------------------------------------------------

def build_ocr(det_model: str, rec_model: str):
    from paddleocr import PaddleOCR
    return PaddleOCR(
        lang="en",
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def ocr_image(ocr, image_path: Path) -> dict:
    """Ejecuta PaddleOCR sobre una imagen y devuelve líneas + texto ordenado."""
    from PIL import Image
    width, height = Image.open(image_path).size

    t0 = time.time()
    result = ocr.predict(str(image_path))[0]
    seconds = time.time() - t0

    lines = [{"text": t, "score": round(float(s), 4), "box": [int(v) for v in b]}
             for t, s, b in zip(result["rec_texts"], result["rec_scores"], result["rec_boxes"])]
    text, n_columns = reading_order(lines, width)
    return {"image_size": [width, height], "seconds": round(seconds, 3),
            "n_columns": n_columns, "n_lines": len(lines),
            "mean_score": round(float(np.mean([l["score"] for l in lines])), 4) if lines else None,
            "text": text, "lines": lines}


def hardware() -> str:
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return f"GPU {paddle.device.cuda.get_device_name(0)}"
    except Exception:
        pass
    return f"CPU {platform.processor() or platform.machine()}"


def summarize(out_dir: Path, failed: list[dict], det_model: str, rec_model: str) -> dict:
    records = [load_json(p) for p in sorted(out_dir.glob("*.json"))]
    seconds = np.array([r["seconds"] for r in records])
    scores = np.array([r["mean_score"] for r in records if r["mean_score"] is not None])
    return {
        "det_model": det_model,
        "rec_model": rec_model,
        "hardware": hardware(),
        "n_processed": len(records),
        "n_failed": len(failed),
        "failed": failed,
        "seconds_per_cv_mean": round(float(seconds.mean()), 3),
        "seconds_per_cv_median": round(float(np.median(seconds)), 3),
        "total_ocr_minutes": round(float(seconds.sum()) / 60, 1),
        "mean_confidence": round(float(scores.mean()), 4),
        "share_lines_below_0.8": round(float(np.mean(
            [l["score"] < 0.8 for r in records for l in r["lines"]])), 4),
        "n_two_column_cvs": int(sum(r["n_columns"] == 2 for r in records)),
        "lines_per_cv_mean": round(float(np.mean([r["n_lines"] for r in records])), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--det-model", default=DET_MODEL)
    parser.add_argument("--rec-model", default=REC_MODEL)
    parser.add_argument("--limit", type=int, help="procesar solo n CVs al azar (benchmark)")
    parser.add_argument("--out-dir", type=Path, default=OCR_DIR)
    parser.add_argument("--image", type=Path, help="OCR de una sola imagen (lo usa la app)")
    parser.add_argument("--json-out", type=Path, help="con --image: dónde guardar el resultado")
    args = parser.parse_args()

    if args.image:
        record = ocr_image(build_ocr(args.det_model, args.rec_model), args.image)
        save_json({"det_model": args.det_model, "rec_model": args.rec_model, **record},
                  args.json_out)
        return

    ids = pd.read_csv(SAMPLED_IDS_CSV, dtype=str)[ID_COL]
    if args.limit:
        ids = ids.sample(args.limit, random_state=SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pending = [cv for cv in ids if not (args.out_dir / f"{cv}.json").exists()]
    print(f"{len(ids)} CVs, {len(ids) - len(pending)} ya en caché, {len(pending)} por procesar. "
          f"Modelos: {args.det_model} + {args.rec_model} | {hardware()}")

    ocr = build_ocr(args.det_model, args.rec_model)
    failed, t0 = [], time.time()
    for i, cv_id in enumerate(pending, 1):
        try:
            record = ocr_image(ocr, RAW_IMG_DIR / f"{cv_id}.png")
            save_json({ID_COL: cv_id, "det_model": args.det_model,
                       "rec_model": args.rec_model, **record},
                      args.out_dir / f"{cv_id}.json")
        except Exception as e:  # una imagen dañada no detiene el lote
            failed.append({ID_COL: cv_id, "error": repr(e)[:300]})
            print(f"FALLÓ {cv_id}: {e!r}")
        if i % 50 == 0 or i == len(pending):
            elapsed = time.time() - t0
            print(f"{i}/{len(pending)} ({elapsed / i:.2f} s/CV, "
                  f"faltan ~{elapsed / i * (len(pending) - i) / 60:.0f} min)")

    summary = summarize(args.out_dir, failed, args.det_model, args.rec_model)
    name = "ocr_summary.json" if args.out_dir == OCR_DIR else f"ocr_summary_{args.out_dir.name}.json"
    save_json(summary, METRICS_DIR / name)
    print(summary)


if __name__ == "__main__":
    main()
