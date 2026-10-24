"""Fase 3 — Limpieza del texto OCR, eliminación de datos personales y enmascarado.

Produce data/processed/cvs.csv con una fila por CV y estas versiones del texto
(todas sin datos personales y con las respuestas directas enmascaradas):
  - text_raw:        texto OCR en orden de lectura, sin más limpieza
  - text_clean:      + reparación de rangos de fechas, guiones/viñetas y líneas cortadas
  - text_clean_conf: como text_clean, pero sin las líneas con confianza < umbral
  - text_ref:        texto de referencia con la misma limpieza (solo para la Pregunta 2,
                     nunca para entrenar)
Las columnas sex y race se conservan únicamente para el chequeo de equidad (Pregunta 4).

Uso:
    python -m src.preprocessing
    python -m src.preprocessing --conf-threshold 0.9
"""

import argparse
import re
import unicodedata

import numpy as np
import pandas as pd

from src.ocr_extraction import reading_order
from src.utils import (EXP_CAP, ID_COL, METADATA_CSV, METRICS_DIR, OCR_DIR,
                       PROCESSED_CSV, REFERENCE_TEXT_CSV, SEED, TABLE_DIR, TARGET,
                       clip_years, load_json, save_json)

CONF_THRESHOLD = 0.9

# --- Fechas ------------------------------------------------------------------
# PopResume escribe los periodos como AAAA.MM–AAAA.MM. El OCR a veces lee el guion
# como "—" o "-", y a veces lo omite ("2010.122014.01"); los dígitos sí son correctos.
DATE_RANGE_RE = re.compile(r"((?:19|20)\d{2}\.\d{2})\s*[-–—~]?\s*((?:19|20)\d{2}\.\d{2})")
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def repair_date_ranges(text: str) -> tuple[str, int]:
    """Normaliza todos los rangos a "AAAA.MM - AAAA.MM" y cuenta los que no tenían guion."""
    n_missing = len(re.findall(r"(?:19|20)\d{2}\.\d{2}(?:19|20)\d{2}\.\d{2}", text))
    return DATE_RANGE_RE.sub(r"\1 - \2", text), n_missing


# --- Datos personales ----------------------------------------------------------
EMAIL_RE = re.compile(r"\S+@\S+\.\w+")
PHONE_RE = re.compile(r"\+?\d{1,2}[\s.-]?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")
LOCATION_RE = re.compile(r"\b[A-Z]{2},\s*USA\b")


SECTION_RE = re.compile(r"^\s*(WORK HISTORY|EXPERIENCE|EDUCATION|SKILLS)\b", re.IGNORECASE | re.MULTILINE)


def remove_pii(text: str, first_name: str, last_name: str) -> tuple[str, dict]:
    """Elimina nombre, email, teléfono y ubicación (estado). El nombre se toma de la
    metadata solo para poder borrarlo (nunca se usa como feature) y se busca solo en
    el encabezado, para no borrar empresas como "Park Medical" si el apellido es Park."""
    counts = {}
    for key, pattern in [("email", EMAIL_RE), ("phone", PHONE_RE), ("location", LOCATION_RE)]:
        text, counts[key] = pattern.subn("", text)
    first_section = SECTION_RE.search(text)
    cut = first_section.start() if first_section else 0
    header, body = text[:cut], text[cut:]
    n_name = 0
    for name in {first_name, last_name}:
        if isinstance(name, str) and name:
            header, n = re.subn(rf"\b{re.escape(name)}\b", "", header, flags=re.IGNORECASE)
            n_name += n
    text = header + body
    counts["name"] = n_name
    # Las líneas que quedan solo con separadores ("|  |") se eliminan
    lines = [l for l in text.split("\n") if not l.strip() or re.search(r"\w", l)]
    return "\n".join(lines).strip(), counts


# --- Respuestas directas ---------------------------------------------------------
# Frases que dan la experiencia total, la edad o la fecha de nacimiento. Las fechas de
# cada trabajo y los años de graduación NO se tocan: son la evidencia del modelo.
NUM = r"(?:\d{1,2}(?:\.\d)?\+?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)"
MASK_PATTERNS = {
    "total_experience": re.compile(
        rf"\b(?:(?:over|more than|nearly|almost|about|around)\s+)?{NUM}\s*(?:\+\s*)?years?"
        rf"(?:\s+of)?\s+(?:\w+\s+){{0,3}}?experience\b", re.IGNORECASE),
    "experience_colon": re.compile(rf"\bexperience\s*[:\-]\s*{NUM}\s*years?\b", re.IGNORECASE),
    "age": re.compile(rf"\b(?:age[d]?\s*[:\-]?\s*\d{{2}}\b|\d{{2}}\s*(?:years?|yrs?)[\s-]*old\b)",
                      re.IGNORECASE),
    "birth": re.compile(r"\b(?:date of birth|d\.?o\.?b\.?|born(?:\s+(?:on|in))?)\s*[:\-]?\s*"
                        r"[\w ,./-]{0,20}?(?:19|20)\d{2}\b", re.IGNORECASE),
}
MASK_TOKEN = "[MASKED]"


def mask_direct_answers(text: str) -> tuple[str, dict]:
    counts = {}
    for key, pattern in MASK_PATTERNS.items():
        text, counts[key] = pattern.subn(MASK_TOKEN, text)
    return text, counts


# --- Limpieza --------------------------------------------------------------------
DASHES_RE = re.compile(r"[‐‑‒–—―−]")
BULLETS_RE = re.compile(r"^\s*[•·▪●◦*]\s*", re.MULTILINE)


def clean_text(text: str) -> tuple[str, int]:
    """Normaliza unicode, guiones y viñetas, repara rangos de fechas, une palabras
    cortadas con guion al final de línea y líneas que continúan la anterior."""
    text = unicodedata.normalize("NFKC", text)
    text = DASHES_RE.sub("-", text)
    text, n_repaired = repair_date_ranges(text)
    text = BULLETS_RE.sub("- ", text)
    text = re.sub(r"^-(?=\S)", "- ", text, flags=re.MULTILINE)  # "-Texto" -> "- Texto"
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)  # hyphenation: "manage-\nment"
    text = re.sub(r"(?<=[a-z,])\n(?=[a-z])", " ", text)  # línea cortada a mitad de frase
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?m)^ ", "", text)  # viñeta perdida por el OCR deja un espacio inicial
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), n_repaired


def text_without_low_conf(record: dict, threshold: float) -> tuple[str, int]:
    """Reconstruye el orden de lectura usando solo las líneas con confianza >= umbral."""
    kept = [l for l in record["lines"] if l["score"] >= threshold]
    text, _ = reading_order(kept, record["image_size"][0])
    return text, len(record["lines"]) - len(kept)


def process_cv(text: str, meta: pd.Series) -> tuple[str, dict, dict]:
    """PII + enmascarado: obligatorio para cualquier versión del texto."""
    text, pii = remove_pii(text, meta["first_name"], meta["last_name"])
    text, masked = mask_direct_answers(text)
    return text, pii, masked


def prepare_text(ocr_text: str) -> str:
    """Limpieza para un CV nuevo (app): sin metadata no conocemos el nombre, así que se
    descarta todo el encabezado anterior a la primera sección (nombre y contacto)."""
    text, _ = remove_pii(ocr_text, None, None)
    first_section = SECTION_RE.search(text)
    if first_section:
        text = text[first_section.start():]
    text, _ = mask_direct_answers(text)
    return clean_text(text)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    meta = pd.read_csv(METADATA_CSV, dtype={ID_COL: str, "id": str})
    ref = pd.read_csv(REFERENCE_TEXT_CSV, dtype=str).set_index(ID_COL)["text"]

    rows, pii_tot, mask_tot = [], {}, {}
    n_repaired = n_low_conf = 0
    missing_ocr = []
    for i, (_, m) in enumerate(meta.iterrows(), 1):
        if i % 500 == 0:
            print(f"{i}/{len(meta)} CVs procesados")
        path = OCR_DIR / f"{m[ID_COL]}.json"
        if not path.exists():
            missing_ocr.append(m[ID_COL])
            continue
        record = load_json(path)

        raw, pii, masked = process_cv(record["text"], m)
        clean, n_rep = clean_text(raw)
        conf_text, n_drop = text_without_low_conf(record, args.conf_threshold)
        conf_clean, _ = clean_text(process_cv(conf_text, m)[0])
        ref_clean, _ = clean_text(process_cv(ref.get(m[ID_COL], ""), m)[0])

        n_repaired += n_rep
        n_low_conf += n_drop
        for k, v in pii.items():
            pii_tot[k] = pii_tot.get(k, 0) + v
        for k, v in masked.items():
            mask_tot.setdefault(k, {"phrases": 0, "cvs": 0})
            mask_tot[k]["phrases"] += v
            mask_tot[k]["cvs"] += int(v > 0)

        rows.append({
            ID_COL: m[ID_COL], "occupation": m["occupation"],
            TARGET: m[TARGET], "y": float(clip_years(m[TARGET])),
            "sex": m["sex"], "race": m["race"],  # solo para la Pregunta 4
            "n_masked": sum(masked.values()), "n_low_conf_lines": n_drop,
            "text_raw": raw, "text_clean": clean,
            "text_clean_conf": conf_clean, "text_ref": ref_clean,
        })

    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED_CSV, index=False)

    # 50 textos al azar para la lectura manual exigida por la tarea
    sample = df.sample(min(50, len(df)), random_state=SEED)
    with open(TABLE_DIR / "review_sample_50.txt", "w", encoding="utf-8") as f:
        for _, r in sample.iterrows():
            f.write(f"===== {r[ID_COL]} | exp_year={r[TARGET]:.2f} =====\n{r['text_clean']}\n\n")

    summary = {
        "n_cvs": len(df),
        "missing_ocr": missing_ocr,
        "conf_threshold": args.conf_threshold,
        "low_conf_lines_dropped": n_low_conf,
        "cvs_with_low_conf_lines": int((df["n_low_conf_lines"] > 0).sum()),
        "date_ranges_missing_dash_repaired": n_repaired,
        "pii_removed": pii_tot,
        "masking": mask_tot,
        "cvs_masked": int((df["n_masked"] > 0).sum()),
        "mean_chars_clean": round(float(df["text_clean"].str.len().mean()), 1),
        "mean_words_clean": round(float(df["text_clean"].str.split().str.len().mean()), 1),
        "cap": EXP_CAP,
    }
    save_json(summary, METRICS_DIR / "preprocessing_summary.json")
    print(summary)
    print(f"\nEjemplo limpio ({df.iloc[0][ID_COL]}):\n{df.iloc[0]['text_clean']}")


if __name__ == "__main__":
    main()
