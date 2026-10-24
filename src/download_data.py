"""Fase 1 — Muestreo estratificado y descarga de CVs desde PopResume.

Estrategia en dos pasos (más rápida y verificable que el streaming con buffer):
  1. Índice: se leen solo las columnas livianas (sin imágenes) de todos los shards y
     se sortean N CVs sin foto por ocupación con una semilla fija. Es un muestreo
     aleatorio uniforme sobre toda la población de CVs sin foto.
  2. Imágenes: se descarga cada shard, se extraen solo las imágenes sorteadas y el
     shard se borra. Si se interrumpe, al reanudar se saltan las imágenes ya guardadas.

Después se descarga la configuración `text` y se conserva solo el texto de referencia
de los CVs muestreados (se usa únicamente para medir el OCR, nunca como feature).

El `id` de PopResume es un correlativo que se repite entre ocupaciones, por eso la
clave del proyecto es `cv_id = "<job>_<id>"`.

Uso:
    python -m src.download_data --inspect        # ver shards, ocupaciones y columnas
    python -m src.download_data                  # muestreo completo (800 por ocupación)
    python -m src.download_data --n-per-occ 400  # versión reducida (2,000 CVs)
"""

import argparse
import io
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

from src.utils import (EXP_CAP, HF_DATASET, ID_COL, METADATA_CSV, METRICS_DIR,
                       N_PER_OCCUPATION, RAW_IMG_DIR, REFERENCE_TEXT_CSV,
                       SAMPLED_IDS_CSV, SEED, TABLE_DIR, TARGET, clip_years,
                       ensure_dirs, save_json)

REPO_PREFIX = f"datasets/{HF_DATASET}/"
IMAGES_GLOB = f"{REPO_PREFIX}data/images/*.parquet"
SHARD_TMP_DIR = Path(tempfile.gettempdir()) / "popresume_shards"  # disco local, no Drive
CAP_WARNING_SHARE = 0.15


def make_cv_id(row) -> str:
    return f"{row['job']}_{row['id']}"


def list_shards() -> dict[str, list[str]]:
    """{ocupación: [rutas de shard]} a partir de los nombres <ocupacion>-NNNN.parquet."""
    shards: dict[str, list[str]] = {}
    for path in sorted(HfFileSystem().glob(IMAGES_GLOB)):
        occupation = path.rsplit("/", 1)[-1].rsplit("-", 1)[0]
        shards.setdefault(occupation, []).append(path)
    return shards


def inspect() -> None:
    shards = list_shards()
    print(f"{sum(map(len, shards.values()))} shards de imágenes, {len(shards)} ocupaciones:")
    for occ, paths in shards.items():
        print(f"  - {occ}: {len(paths)} shards")

    first = next(iter(shards.values()))[0]
    schema = pq.ParquetFile(HfFileSystem().open(first)).schema_arrow
    print(f"\nColumnas de la config de imágenes: {schema.names}")

    text_ds = load_dataset(HF_DATASET, "text", streaming=True)
    split = next(iter(text_ds))
    row = next(iter(text_ds[split]))
    print(f"\nColumnas de la config text (split '{split}'):")
    for k, v in row.items():
        print(f"  {k}: {repr(v)[:120]}")


def read_index(shards: dict[str, list[str]]) -> pd.DataFrame:
    """Lee todas las columnas excepto la imagen de cada shard (solo unos pocos MB)."""
    fs = HfFileSystem()
    frames = []
    for occupation, paths in shards.items():
        n_rows = 0
        for path in paths:
            cols = [c for c in pq.ParquetFile(fs.open(path)).schema_arrow.names if c != "image"]
            df = pq.read_table(path, columns=cols, filesystem=fs).to_pandas()
            df["occupation"] = occupation
            df["shard"] = path.removeprefix(REPO_PREFIX)
            frames.append(df)
            n_rows += len(df)
        print(f"[índice] {occupation}: {n_rows} imágenes en {len(paths)} shards")
    index = pd.concat(frames, ignore_index=True)
    index["id"] = index["id"].astype(str)
    index[ID_COL] = index["job"] + "_" + index["id"]
    return index


def draw_sample(index: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Sortea n CVs sin foto por ocupación. Si un CV tiene varias imágenes sin foto,
    se queda la primera por file_name (regla determinista) y se reporta cuántas había."""
    no_photo = index[~index["has_photo"].astype(bool)].sort_values("file_name")
    variants = no_photo.groupby(ID_COL).size()
    unique = no_photo.drop_duplicates(ID_COL)

    rng = np.random.default_rng(seed)
    parts, population = [], {}
    for occupation, group in unique.groupby("occupation"):
        group = group.sort_values(ID_COL)
        population[occupation] = len(group)
        take = min(n, len(group))
        if take < n:
            print(f"[{occupation}] AVISO: solo hay {len(group)} CVs sin foto.")
        parts.append(group.iloc[np.sort(rng.choice(len(group), size=take, replace=False))])

    sample = pd.concat(parts, ignore_index=True)
    info = {
        "no_photo_population": population,
        "cvs_with_multiple_no_photo_images": int((variants > 1).sum()),
    }
    return sample, info


def download_images(sample: pd.DataFrame) -> None:
    """Descarga shard por shard y guarda solo las imágenes sorteadas como PNG."""
    SHARD_TMP_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(sample["shard"].unique())
    t0 = time.time()
    for i, shard in enumerate(shards, 1):
        rows = sample[sample["shard"] == shard]
        pending = {fn: cv for fn, cv in zip(rows["file_name"], rows[ID_COL])
                   if not (RAW_IMG_DIR / f"{cv}.png").exists()}
        if not pending:
            print(f"[{i}/{len(shards)}] {shard}: ya descargado, se salta.")
            continue

        local = hf_hub_download(HF_DATASET, shard, repo_type="dataset", local_dir=SHARD_TMP_DIR)
        saved = 0
        for batch in pq.ParquetFile(local).iter_batches(columns=["file_name", "image"],
                                                         batch_size=256):
            for fn, img in zip(batch.column("file_name").to_pylist(),
                               batch.column("image").to_pylist()):
                if fn in pending:
                    Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(
                        RAW_IMG_DIR / f"{pending[fn]}.png")
                    saved += 1
        os.remove(local)
        print(f"[{i}/{len(shards)}] {shard}: {saved}/{len(pending)} imágenes guardadas "
              f"({(time.time() - t0) / 60:.1f} min acumulados)")

    missing = [cv for cv in sample[ID_COL] if not (RAW_IMG_DIR / f"{cv}.png").exists()]
    if missing:
        print(f"AVISO: faltan {len(missing)} imágenes, p. ej. {missing[:3]}")


def save_reference_text(meta: pd.DataFrame) -> None:
    """Guarda cv_id + texto de referencia de los CVs muestreados (solo para CER/WER)."""
    text_ds = load_dataset(HF_DATASET, "text")
    text_ds = text_ds[next(iter(text_ds))]
    wanted = set(meta[ID_COL])
    text_df = (text_ds.select_columns(["id", "job", "text"])
               .filter(lambda r: make_cv_id(r) in wanted)
               .to_pandas())
    text_df[ID_COL] = text_df["job"] + "_" + text_df["id"].astype(str)
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
    args = parser.parse_args()

    if args.inspect:
        inspect()
        return

    ensure_dirs()
    t0 = time.time()
    shards = list_shards()
    print(f"Ocupaciones: {list(shards)}")

    index = read_index(shards)
    meta, sample_info = draw_sample(index, args.n_per_occ, args.seed)
    meta[f"{TARGET}_capped"] = clip_years(meta[TARGET])
    meta.to_csv(METADATA_CSV, index=False)
    meta[[ID_COL, "id", "occupation", "file_name"]].to_csv(SAMPLED_IDS_CSV, index=False)
    print(f"Muestra sorteada: {len(meta)} CVs. IDs en {SAMPLED_IDS_CSV}")

    download_images(meta)
    save_reference_text(meta)

    summary = summarize_target(meta)
    summary.update({
        "seed": args.seed,
        "n_per_occupation": args.n_per_occ,
        "per_occupation": meta["occupation"].value_counts().to_dict(),
        **sample_info,
        "download_minutes": round((time.time() - t0) / 60, 1),
    })
    save_json(summary, METRICS_DIR / "sampling_summary.json")
    print(f"\nListo: {len(meta)} CVs en {summary['download_minutes']} min.")


if __name__ == "__main__":
    main()
