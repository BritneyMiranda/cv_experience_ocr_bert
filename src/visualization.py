"""Figuras del proyecto (output/figures/), leídas por el README y la app.

Colores: un color fijo por modelo en todas las figuras (paleta categórica de referencia,
slots 1–4 en su orden validado; la mediana en gris por ser la línea base). Cada figura
tiene su tabla en output/tables/ con los mismos números.

Uso:
    python -m src.visualization
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation import BAND_ORDER, MAIN_MODELS, load_test
from src.utils import EXP_CAP, FIG_DIR, PROCESSED_CSV, TABLE_DIR

# Orden de dibujo = orden de la paleta (azul, naranja, aqua, amarillo) + gris
ORDER = ["Fine-tuned BERT", "TF-IDF + Ridge", "Embeddings + Ridge", "Rule-based (bonus)", "Median"]
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#898781"]))
NAME_BY_LABEL = {v: k for k, v in MAIN_MODELS.items()}
OCC_SHORT = {"accountants_auditors": "Accountants", "construction_laborers": "Construction",
             "elementary_middle_school_teachers": "Teachers",
             "registered_nurses": "Nurses", "software_developers": "Software dev."}

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "font.size": 10,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


def target_distribution() -> None:
    cvs = pd.read_csv(PROCESSED_CSV, usecols=["occupation", "exp_year"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    a1.hist(cvs["exp_year"], bins=np.arange(0, 28, 1), color=COLORS["Fine-tuned BERT"],
            edgecolor=SURFACE, linewidth=1)
    a1.axvline(EXP_CAP, color=INK, linewidth=1, linestyle="--")
    n_cap = int((cvs["exp_year"] >= EXP_CAP).sum())
    a1.annotate(f"tope = 20 años\n{n_cap} CVs ({n_cap / len(cvs):.1%}) ≥ 20", (EXP_CAP, a1.get_ylim()[1] * 0.8),
                xytext=(6, 0), textcoords="offset points", color=INK, fontsize=9)
    a1.set(title="Años de experiencia (4,000 CVs)", xlabel="exp_year", ylabel="CVs")
    groups = [cvs.loc[cvs["occupation"] == o, "exp_year"] for o in OCC_SHORT]
    a2.boxplot(groups, orientation="horizontal", tick_labels=list(OCC_SHORT.values()), widths=0.5,
               medianprops={"color": INK}, flierprops={"markersize": 3, "markeredgecolor": MUTED})
    a2.set(title="Por ocupación", xlabel="exp_year")
    a2.grid(axis="y", visible=False)
    save(fig, "target_distribution")


def ocr_quality() -> None:
    per_cv = pd.read_csv(TABLE_DIR / "ocr_quality_per_cv.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.hist(per_cv["cer"] * 100, bins=40, color=COLORS["Fine-tuned BERT"], edgecolor=SURFACE)
    a1.axvline(per_cv["cer"].mean() * 100, color=INK, linestyle="--", linewidth=1)
    a1.set(title="CER por CV (texto normalizado)", xlabel="CER (%)", ylabel="CVs")
    by_occ = per_cv.groupby("occupation")[["years_ok_raw", "years_ok_repaired"]].mean() * 100
    by_occ.index = [OCC_SHORT[o] for o in by_occ.index]
    y = np.arange(len(by_occ))
    a2.barh(y + 0.2, by_occ["years_ok_raw"], height=0.38, color="#86b6ef", label="OCR crudo")
    a2.barh(y - 0.2, by_occ["years_ok_repaired"], height=0.38, color="#1c5cab",
            label="tras reparar rangos")
    a2.set_yticks(y, by_occ.index)
    a2.set(title="Años de la referencia leídos correctamente", xlabel="%", xlim=(80, 100.5))
    a2.legend(loc="lower right")
    a2.grid(axis="y", visible=False)
    save(fig, "ocr_quality")


def pred_vs_true() -> None:
    fig, axes = plt.subplots(1, len(ORDER), figsize=(17, 4.4), sharex=True, sharey=True)
    for ax, label in zip(axes, ORDER):
        df = load_test(NAME_BY_LABEL[label])
        ax.plot([0, EXP_CAP], [0, EXP_CAP], color=INK, linewidth=1, linestyle="--")
        ax.scatter(df["y_true"], df["y_pred"], s=9, alpha=0.5, color=COLORS[label],
                   edgecolors="none")
        ax.set_title(f"{label}\nMAE {df['abs_err'].mean():.2f}", fontsize=10)
        ax.set_xlabel("años reales")
        ax.set_aspect("equal")
    axes[0].set_ylabel("años predichos")
    axes[0].set_xlim(0, EXP_CAP + 0.5)
    axes[0].set_ylim(0, EXP_CAP + 0.5)
    save(fig, "pred_vs_true_all")
    # Una figura por modelo para el selector de la app
    for label in ORDER:
        df = load_test(NAME_BY_LABEL[label])
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        ax.plot([0, EXP_CAP], [0, EXP_CAP], color=INK, linewidth=1, linestyle="--", label="45°")
        ax.scatter(df["y_true"], df["y_pred"], s=12, alpha=0.5, color=COLORS[label],
                   edgecolors="none")
        ax.set(title=f"{label} (test, MAE {df['abs_err'].mean():.2f})", xlabel="años reales",
               ylabel="años predichos", xlim=(0, EXP_CAP + 0.5), ylim=(0, EXP_CAP + 0.5))
        ax.set_aspect("equal")
        save(fig, f"pred_vs_true_{NAME_BY_LABEL[label]}")


def grouped_bars(table: str, col: str, value: str, name: str, title: str, ylabel: str,
                 categories: list, tick_labels: list, models: list = ORDER) -> None:
    df = pd.read_csv(TABLE_DIR / table)
    piv = df.pivot(index=col, columns="model", values=value).reindex(categories)[models]
    x = np.arange(len(categories))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, label in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width, piv[label], width=width * 0.92,
               color=COLORS[label], label=label)
    if value == "mean_error":
        ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, tick_labels)
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
    save(fig, name)


def confusions() -> None:
    for name, label in MAIN_MODELS.items():
        cm = pd.read_csv(TABLE_DIR / f"band_confusion_{name}.csv", index_col=0)
        share = cm.div(cm.sum(axis=1), axis=0)  # color = % de cada banda real
        fig, ax = plt.subplots(figsize=(4.4, 3.9))
        sns.heatmap(share, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    linewidths=2, linecolor=SURFACE, annot_kws={"fontsize": 10}, ax=ax)
        ax.set(title=label, xlabel="banda predicha", ylabel="banda real")
        ax.tick_params(axis="y", rotation=0)
        save(fig, f"confusion_{name}")


def bert_gain() -> None:
    g = pd.read_csv(TABLE_DIR / "bert_gain_bootstrap.csv")
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    y = np.arange(len(g))[::-1]
    gain = g["MAE gain (baseline - BERT)"]
    ax.errorbar(gain, y, xerr=[gain - g["CI 95% low"], g["CI 95% high"] - gain], fmt="o",
                color=COLORS["Fine-tuned BERT"], ecolor=INK, capsize=4, markersize=7)
    ax.axvline(0, color=INK, linewidth=0.8)
    for yi, v, lo, hi in zip(y, gain, g["CI 95% low"], g["CI 95% high"]):
        ax.annotate(f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]", (v, yi), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [f"vs {b}" for b in g["baseline"]])
    ax.set(title="Reducción de MAE de BERT en test (IC 95 %, bootstrap pareado)",
           xlabel="años de MAE que BERT ahorra (negativo = BERT es peor)")
    ax.set_ylim(-0.6, len(g) - 0.3)
    ax.grid(axis="y", visible=False)
    save(fig, "bert_gain_ci")


def fairness() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [2, 3]})
    for ax, col in zip(axes, ["sex", "race"]):
        f = pd.read_csv(TABLE_DIR / f"q4_fairness_{col}.csv")
        y = np.arange(len(f))[::-1]
        me = f["mean_error (pred - true)"]
        ax.errorbar(me, y, xerr=[me - f["CI 95% low"], f["CI 95% high"] - me], fmt="o",
                    color=COLORS["Fine-tuned BERT"], ecolor=INK, capsize=4)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_yticks(y, [f"{g} (n={n})" for g, n in zip(f[col], f["n"])])
        ax.set(title=f"BERT: error medio por {'sexo' if col == 'sex' else 'raza'}",
               xlabel="predicho − real (años), IC 95 %")
        ax.grid(axis="y", visible=False)
    save(fig, "fairness_bert")


def probe() -> None:
    path = TABLE_DIR / "q1_probe_summary.csv"
    if not path.exists():  # se genera con src/probe.py (necesita los modelos)
        return
    s = pd.read_csv(path)
    changes = {"delete_job": "Borrar el trabajo\nmás antiguo", "shift_-5": "Todas las fechas\n−5 años",
               "shift_+5": "Todas las fechas\n+5 años"}
    models = ORDER[:4]
    x = np.arange(len(changes))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, label in enumerate(models):
        vals = s[s["model"] == label].set_index("change").loc[list(changes), "mean_delta_pred"]
        ax.bar(x + (i - 1.5) * width, vals, width=width * 0.92, color=COLORS[label], label=label)
    expected = s.drop_duplicates("change").set_index("change").loc[list(changes),
                                                                   "mean_delta_expected"]
    ax.scatter(x, expected, marker="_", s=2500, color=INK, linewidths=2, zorder=3,
               label="lo que haría un reclutador")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, list(changes.values()))
    ax.set(title="Prueba de fechas: cambio medio de la predicción (20 CVs de test)",
           ylabel="Δ predicción (años)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=9,
              markerscale=0.2)
    save(fig, "q1_probe")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    probe()
    target_distribution()
    ocr_quality()
    pred_vs_true()
    grouped_bars("mae_by_band.csv", "band", "MAE", "mae_by_band",
                 "MAE por banda de experiencia real (test)", "MAE (años)", BAND_ORDER,
                 [f"{b} años" for b in BAND_ORDER])
    grouped_bars("mae_by_band.csv", "band", "mean_error", "mean_error_by_band",
                 "Error medio por banda: ¿tiran hacia el centro? (sin la mediana)",
                 "predicho − real (años)", BAND_ORDER, [f"{b} años" for b in BAND_ORDER],
                 models=ORDER[:4])
    occs = list(OCC_SHORT)
    grouped_bars("mae_by_occupation.csv", "occupation", "MAE", "mae_by_occupation",
                 "MAE por ocupación (test)", "MAE (años)", occs, [OCC_SHORT[o] for o in occs])
    confusions()
    bert_gain()
    fairness()
    print(f"Figuras en {FIG_DIR}: {sorted(p.name for p in FIG_DIR.glob('*.png'))}")


if __name__ == "__main__":
    main()
