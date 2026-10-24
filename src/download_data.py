"""Fase 1 — Muestreo estratificado y descarga de CVs desde PopResume.

Para cada ocupación se leen en streaming los shards de imágenes, se filtran los CVs
sin foto, se barajan con semilla fija y se guardan los primeros N como PNG. Luego se
descarga la configuración `text` y se conserva solo el texto de referencia de los CVs
muestreados (se usa únicamente para medir la calidad del OCR, nunca como feature).

El `id` de PopResume es un correlativo que se repite entre ocupaciones, por eso la
clave del proyecto es `cv_id = "<job>_<id>"`.

Uso:
    python -m src.download_data --inspect        # ver shards, ocupaciones y columnas
    python -m src.download_data                  # muestreo completo (800 por ocupación)
    python -m src.download_data --n-per-occ 400  # versión reducida (2,000 CVs)
"""

import argparse
import io
import time

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from PIL import Image

from src.utils import (EXP_CAP, HF_DATASET, ID_COL, METADATA_CSV, METRICS_DIR,
                       N_PER_OCCUPATION, RAW_DIR, RAW_IMG_DIR, REFERENCE_TEXT_CSV,
                       SAMPLED_IDS_CSV, SEED, TABLE_DIR, TARGET, clip_years,
                       ensure_dirs, save_json)

IMAGES_GLOB = f"datasets/{HF_DATASET}/data/images/*.parquet"
CAP_WARNING_SHARE = 0.15


def make_cv_id(row) -> str:
    return f"{row['job']}_{row['id']}"


def list_occupations() -> list[str]:
    """Deduce las ocupaciones a partir de los nombres de shard <ocupacion>-NNNN.parquet."""
    shards = HfFileSystem().glob(IMAGES_GLOB)
    return sorted({s.rsplit("/", 1)[-1].rsplit("-", 1)[0] for s in shards})


def stream_occupation(occupation: str, seed: int, buffer_size: int):
    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{HF_DATASET}/data/images/{occupation}-*.parquet",
        split="train",
        streaming=True,
    )
    return ds.filter(lambda r: not r["has_photo"]).shuffle(seed=seed, buffer_size=buffer_size)


def to_pil(image) -> Image.Image:
    """El campo image puede llegar decodificado (PIL) o como {'bytes', 'path'}."""
    if isinstance(image, Image.Image):
        return image
    return Image.open(io.BytesIO(image["bytes"]))


def inspect() -> None:
    shards = HfFileSystem().glob(IMAGES_GLOB)
    occupations = list_occupations()
    print(f"{len(shards)} shards de imágenes, {len(occupations)} ocupaciones:")
    for occ in occupations:
        print(f"  - {occ}: {sum(s.rsplit('/', 1)[-1].startswith(occ + '-') for s in shards)} shards")

    row = next(iter(stream_occupation(occupations[0], SEED, buffer_size=1)))
    print("\nColumnas de la config de imágenes (primer CV sin foto):")
    for k, v in row.items():
        shown = type(v).__name__ if k == "image" else repr(v)[:120]
        print(f"  {k}: {shown}")

    text_ds = load_dataset(HF_DATASET, "text", streaming=True)
    split = next(iter(text_ds))
    row = next(iter(text_ds[split]))
    print(f"\nColumnas de la config text (split '{split}'):")
    for k, v in row.items():
        print(f"  {k}: {repr(v)[:120]}")


def sample_occupation(occupation: str, n: int, seed: int, buffer_size: int) -> pd.DataFrame:
    """Guarda n imágenes únicas de una ocupación y devuelve su metadata (sin la imagen).

    Si la ocupación ya fue descargada completa, reutiliza su CSV (permite reanudar).
    """
    occ_csv = RAW_DIR / f"metadata_{occupation}.csv"
    if occ_csv.exists():
        cached = pd.read_csv(occ_csv, dtype={ID_COL: str, "id": str})
        if len(cached) == n and all((RAW_IMG_DIR / f"{i}.png").exists() for i in cached[ID_COL]):
            print(f"[{occupation}] ya descargada ({n} CVs), se reutiliza.")
            return cached

    rows, n_duplicates, t0 = [], 0, time.time()
    seen = set()
    for r in stream_occupation(occupation, seed, buffer_size):
        cv_id = make_cv_id(r)
        if cv_id in seen:  # otra variante de imagen del mismo CV
            n_duplicates += 1
            continue
        seen.add(cv_id)
        to_pil(r["image"]).convert("RGB").save(RAW_IMG_DIR / f"{cv_id}.png")
        meta = {k: v for k, v in r.items() if k != "image"}
        rows.append({ID_COL: cv_id, "occupation": occupation, **meta})
        if len(rows) % 100 == 0:
            print(f"[{occupation}] {len(rows)}/{n} ({time.time() - t0:.0f} s)")
        if len(rows) == n:
            break

    if len(rows) < n:
        print(f"[{occupation}] AVISO: solo había {len(rows)} CVs sin foto.")
    if n_duplicates:
        print(f"[{occupation}] {n_duplicates} imágenes duplicadas del mismo CV descartadas.")
    df = pd.DataFrame(rows)
    df.to_csv(occ_csv, index=False)
    return df


def save_reference_text(meta: pd.DataFrame) -> None:
    """Guarda cv_id + texto de referencia de los CVs muestreados (solo para CER/WER)."""
    text_ds = load_dataset(HF_DATASET, "text")
    text_ds = text_ds[next(iter(text_ds))]
    wanted = set(meta[ID_COL])
    text_df = (text_ds.select_columns(["id", "job", "text"])
               .filter(lambda r: make_cv_id(r) in wanted)
               .to_pandas())
    text_df[ID_COL] = text_df["job"] + "_" + text_df["id"]
    text_df = text_df.drop_duplicates(ID_COL)

    missing = wanted - set(text_df[ID_COL])
    if missing:
        print(f"AVISO: {len(missing)} CVs sin texto de referencia.")
    text_df[[ID_COL, "text"]].to_csv(REFERENCE_TEXT_CSV, index=False)
    print(f"Texto de referencia de {len(text_df)} CVs en {REFERENCE_TEXT_CSV}")


def summarize_target(meta: pd.DataFrame) -> dict:
    """Distribución de exp_year y revisión del tope (regla del 15 %)."""
    raw = meta[TARGET].astype(float)
    at_cap = raw >= EXP_CAP

    def stats(g: pd.DataFrame) -> pd.Series:
        y = g[TARGET].astype(float)
        return pd.Series({
            "n": len(y), "mean": y.mean(), "median": y.median(), "std": y.std(),
            "min": y.min(), "max": y.max(),
            f"n_at_or_above_{EXP_CAP}": int((y >= EXP_CAP).sum()),
            f"share_at_or_above_{EXP_CAP}": (y >= EXP_CAP).mean(),
        })

    table = meta.groupby("occupation").apply(stats, include_groups=False)
    table.loc["ALL"] = stats(meta)
    table.round(3).to_csv(TABLE_DIR / "target_distribution.csv")
    print("\nDistribución de exp_year (sin recortar):")
    print(table.round(2).to_string())

    share = float(at_cap.mean())
    summary = {
        "n_cvs": len(meta),
        "cap": EXP_CAP,
        "n_at_or_above_cap": int(at_cap.sum()),
        "share_at_or_above_cap": share,
        "n_above_cap_clipped": int((raw > EXP_CAP).sum()),
        "cap_rule_triggered": share > CAP_WARNING_SHARE,
    }
    msg = f"{summary['n_at_or_above_cap']} CVs ({share:.1%}) quedan en el tope de {EXP_CAP} años."
    if summary["cap_rule_triggered"]:
        msg += " Supera el 15 %: evaluar subir el tope a 25 o 30 y justificarlo."
    print("\n" + msg)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--n-per-occ", type=int, default=N_PER_OCCUPATION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--occupations", nargs="*", help="por defecto, todas las del dataset")
    args = parser.parse_args()

    if args.inspect:
        inspect()
        return

    ensure_dirs()
    occupations = args.occupations or list_occupations()
    print(f"Ocupaciones: {occupations}")

    t0 = time.time()
    parts = [sample_occupation(o, args.n_per_occ, args.seed, args.buffer_size)
             for o in occupations]
    meta = pd.concat(parts, ignore_index=True)
    meta[f"{TARGET}_capped"] = clip_years(meta[TARGET])
    meta.to_csv(METADATA_CSV, index=False)
    meta[[ID_COL, "id", "occupation", "file_name"]].to_csv(SAMPLED_IDS_CSV, index=False)

    save_reference_text(meta)

    summary = summarize_target(meta)
    summary.update({
        "seed": args.seed,
        "buffer_size": args.buffer_size,
        "n_per_occupation": args.n_per_occ,
        "per_occupation": meta["occupation"].value_counts().to_dict(),
        "download_minutes": round((time.time() - t0) / 60, 1),
    })
    save_json(summary, METRICS_DIR / "sampling_summary.json")
    print(f"\nListo: {len(meta)} CVs en {summary['download_minutes']} min. "
          f"IDs en {SAMPLED_IDS_CSV}")


if __name__ == "__main__":
    main()
