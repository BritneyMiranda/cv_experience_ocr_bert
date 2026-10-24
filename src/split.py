"""Fase 4 — Partición train / validation / test (70 / 15 / 15) por CV.

Se estratifica por ocupación × banda de experiencia (0-2, 3-5, 6-10, 11-20), así los
tres conjuntos tienen la misma mezcla de ocupaciones y de niveles de experiencia. La
partición se guarda en data/splits/ y la usan todos los modelos. El test no se mira
hasta la evaluación final: toda la selección de hiperparámetros usa validation.

Uso:
    python -m src.split
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (ID_COL, METRICS_DIR, PROCESSED_CSV, SEED, SPLITS_DIR, TARGET,
                       save_json, to_band)


def main() -> None:
    cvs = pd.read_csv(PROCESSED_CSV, dtype={ID_COL: str})
    cvs["band"] = to_band(cvs["y"])
    strata = cvs["occupation"] + "|" + cvs["band"]

    train, rest = train_test_split(cvs, test_size=0.30, stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=strata.loc[rest.index],
                                 random_state=SEED)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cols = [ID_COL, "occupation", "band", TARGET, "y"]
    summary = {"seed": SEED, "stratify": "occupation x band"}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.sort_values(ID_COL)[cols].to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        summary[name] = {
            "n": len(df),
            "y_mean": round(float(df["y"].mean()), 3),
            "bands": df["band"].value_counts().sort_index().to_dict(),
            "occupations": df["occupation"].value_counts().sort_index().to_dict(),
        }
        print(f"{name:5}: {len(df):4} CVs | media {df['y'].mean():.2f} años | "
              f"bandas {summary[name]['bands']}")

    all_ids = pd.concat([train[ID_COL], val[ID_COL], test[ID_COL]])
    assert all_ids.is_unique and len(all_ids) == len(cvs), "la partición no es disjunta"
    save_json(summary, METRICS_DIR / "split_summary.json")


if __name__ == "__main__":
    main()
