"""App de Streamlit: años de experiencia a partir de imágenes de CVs (4 pestañas).

    streamlit run app.py

Tablas y figuras se leen del repositorio (output/, data/processed, data/splits). Las
imágenes, el OCR cacheado y los modelos se leen de CV_STORAGE_DIR (Google Drive en
Colab). Para el OCR de CVs subidos se usa el Python de OCR_PYTHON (el entorno aislado de
PaddleOCR en Colab); por defecto, el mismo Python de la app.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from src.baseline_rules import periods
from src.predict import LABELS, load_predictors
from src.preprocessing import prepare_text
from src.utils import EXP_CAP, OCR_DIR, RAW_IMG_DIR, ROOT

TABLES = ROOT / "output" / "tables"
FIGS = ROOT / "output" / "figures"
PREDS = ROOT / "output" / "metrics" / "predictions"
MODEL_ORDER = ["bert", "tfidf", "embeddings", "rules", "median"]
OCC_LABELS = {"accountants_auditors": "Accountants & auditors",
              "construction_laborers": "Construction laborers",
              "elementary_middle_school_teachers": "Elementary/middle school teachers",
              "registered_nurses": "Registered nurses",
              "software_developers": "Software developers"}

st.set_page_config(page_title="Experiencia desde CVs", page_icon=":material/badge:",
                   layout="wide")


# --- Datos (cacheados) -----------------------------------------------------------------

@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


@st.cache_data
def test_predictions() -> pd.DataFrame:
    """Una fila por CV de test con la predicción de cada modelo y el texto limpio."""
    cvs = pd.read_csv(ROOT / "data" / "processed" / "cvs.csv", dtype={"cv_id": str})
    test_ids = pd.read_csv(ROOT / "data" / "splits" / "test.csv", dtype={"cv_id": str})["cv_id"]
    df = cvs[cvs["cv_id"].isin(test_ids)][["cv_id", "occupation", "y", "text_clean"]]
    for name in MODEL_ORDER:
        p = pd.read_csv(PREDS / f"{name}.csv", dtype={"cv_id": str})
        p = p[p["split"] == "test"][["cv_id", "y_pred"]].rename(columns={"y_pred": name})
        df = df.merge(p, on="cv_id")
    df["bert_error"] = df["bert"] - df["y"]
    return df.reset_index(drop=True)


def local_examples() -> list[str]:
    """CVs con imagen y OCR disponibles localmente (en Drive dentro de Colab)."""
    if not RAW_IMG_DIR.exists():
        return []
    return sorted(p.stem for p in RAW_IMG_DIR.glob("*.png") if (OCR_DIR / f"{p.stem}.json").exists())


def draw_boxes(image: Image.Image, lines: list[dict]) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for line in lines:
        draw.rectangle(line["box"], outline=(42, 120, 214), width=2)
    return img


@st.cache_resource(show_spinner="Cargando los modelos…")
def predictors() -> dict:
    return load_predictors()


@st.cache_data(show_spinner="Ejecutando PaddleOCR…", max_entries=20)
def run_ocr(image_bytes: bytes) -> dict:
    """OCR de una imagen nueva en un subproceso con el Python del entorno de PaddleOCR."""
    with tempfile.TemporaryDirectory() as tmp:
        img_path, out_path = Path(tmp) / "cv.png", Path(tmp) / "cv.json"
        img_path.write_bytes(image_bytes)
        cmd = [os.environ.get("OCR_PYTHON", sys.executable), "-m", "src.ocr_extraction",
               "--image", str(img_path), "--json-out", str(out_path)]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])
        return json.loads(out_path.read_text(encoding="utf-8"))


def to_png_bytes(uploaded) -> bytes:
    """Imagen subida, o la primera página de un PDF convertida a imagen."""
    data = uploaded.getvalue()
    if uploaded.name.lower().endswith(".pdf"):
        import fitz
        page = fitz.open(stream=data, filetype="pdf")[0]
        return page.get_pixmap(dpi=150).tobytes("png")
    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def show_predictions(text: str) -> None:
    preds = {name: float(fn([text])[0]) for name, fn in predictors().items()}
    cols = st.columns(len(MODEL_ORDER))
    for col, name in zip(cols, MODEL_ORDER):
        value = f"{preds[name]:.1f} años" if name in preds else "no disponible"
        col.metric(LABELS[name], value, border=True)
    found = periods(text)
    if found:
        rows = [{"inicio": f"{s // 12}.{s % 12 + 1:02d}", "fin": f"{e // 12}.{e % 12 + 1:02d}",
                 "años": round((e - s) / 12, 2)} for s, e in found]
        with st.expander("Periodos que encontró el modelo de reglas"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Periodos de trabajo detectados")


# --- Página ------------------------------------------------------------------------------

st.title("¿Cuántos años de experiencia tiene este CV?")
st.caption("PaddleOCR + BERT sobre 4,000 CVs sintéticos de PopResume · 5 ocupaciones · "
           "experiencia de 0 a 20 años")

tab1, tab2, tab3, tab4 = st.tabs([
    ":material/menu_book: Problema y metodología",
    ":material/document_scanner: OCR y exploración",
    ":material/leaderboard: Resultados de los modelos",
    ":material/calculate: Estimador interactivo",
])

# --- Tab 1 -------------------------------------------------------------------------------
with tab1:
    comp = table("model_comparison.csv").set_index("Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CVs analizados", "4,000", border=True)
    c2.metric("CER del OCR", "0.76 %", border=True)
    c3.metric("MAE de BERT (test)", f"{comp.loc['Fine-tuned BERT', 'MAE (years)']:.2f} años",
              border=True)
    c4.metric("MAE de la regla de fechas", f"{comp.loc['Rule-based (bonus)', 'MAE (years)']:.2f} años",
              border=True)

    st.subheader("Objetivo")
    st.markdown(
        "Leer la **imagen** de un CV y estimar los **años de experiencia laboral** del candidato. "
        "La cifra casi nunca está escrita: hay que deducirla conectando las fechas de cada "
        "trabajo, cuántos puestos hay, cómo progresan los cargos y cuándo terminó la formación, "
        "como hace un reclutador. Un modelo de palabras clave no puede hacerlo.")

    st.subheader("Datos y target")
    st.markdown(
        "- **PopResume** (Yu, Park & Moon, EMNLP 2026; CC BY 4.0): CVs sintéticos generados con "
        "estadísticas de población de EE. UU.; ningún dato es de una persona real.\n"
        "- **Muestra:** 800 CVs **sin foto** por ocupación (4,000), sorteados con semilla 42 "
        "sobre toda la población sin foto. Los IDs están en `data/raw/sampled_ids.csv`.\n"
        "- **Target:** `exp_year`, recortado a **0–20 años** (20 = 20 o más). Solo el 0.8 % de "
        "los CVs llega al tope, así que se mantuvo en 20.\n"
        "- **Prohibido como feature:** edad, sexo, raza, nombres, estado y región. Sexo y raza "
        "se usan solo para el chequeo de equidad.\n"
        "- **Regla de enmascarado:** se buscan con regex las frases que dan la respuesta "
        "(\"12 years of experience\", edad, fecha de nacimiento). Se leyeron 50 CVs al azar: "
        "**0 CVs** contenían esas frases. Las fechas de cada trabajo y de graduación no se "
        "tocan, porque son la evidencia.")

    st.subheader("OCR y orden de lectura")
    st.markdown(
        "- **PaddleOCR 3.2** con `PP-OCRv5_server_det` + `en_PP-OCRv5_mobile_rec`. Se eligió el "
        "modelo móvil de inglés frente al `server_rec` multilingüe: tiene el mismo CER, menos "
        "errores de palabra y es 25 % más rápido. 0.71 s por CV en una GPU T4.\n"
        "- **Orden de lectura:** las cajas a la misma altura se unen en una fila, para no separar "
        "un cargo de sus fechas. Si hay un pasillo vertical vacío con texto (no solo fechas) a "
        "ambos lados, se lee primero la columna izquierda y luego la derecha. Todos los CVs de "
        "PopResume usan una sola columna.\n"
        "- **Limpieza:** se reparan los rangos de fechas sin guion (`2017.072020.04` → "
        "`2017.07 - 2020.04`, en el 28 % de los CVs), se normalizan guiones y viñetas, y se "
        "eliminan nombre, email, teléfono y ubicación.")

    st.subheader("Cuatro modelos, una partición")
    st.markdown(
        "Partición **70/15/15 por CV**, estratificada por ocupación × banda de experiencia "
        "(2,800 / 600 / 600). Todos los modelos usan la misma partición; los hiperparámetros se "
        "eligen en validación y el test se evalúa una sola vez.\n\n"
        "| Modelo | Intuición |\n|---|---|\n"
        "| Mediana | Siempre responde lo mismo (5.6 años). Es el piso. |\n"
        "| TF-IDF + Ridge | Cuenta palabras y años sueltos (`2008` → más experiencia), pero no puede restar fechas. |\n"
        "| Embeddings + Ridge | Resume el *tema* del CV con MiniLM congelado; casi ignora los dígitos. |\n"
        "| **DistilBERT ajustado** | Lee el texto completo (≤ 159 tokens) y aprende a combinar las fechas. |\n"
        "| Reglas (bonus) | Extrae los periodos con regex y suma los meses. |")

    st.subheader("Limitaciones")
    st.markdown(
        "CVs **sintéticos** con una sola plantilla; `exp_year` es exactamente la suma de los "
        "periodos y todos los trabajos actuales terminan en 2023.01. Solo hay 5 ocupaciones, con "
        "estadísticas de EE. UU. y en inglés. BERT aprendió a usar 2023 como \"hoy\" (ver "
        "Resultados). Es un dato de apoyo para prellenar perfiles: **nunca debe decidir a quién "
        "se entrevista o se contrata**.")

# --- Tab 2 -------------------------------------------------------------------------------
with tab2:
    st.subheader("Distribución de los años de experiencia")
    st.image(str(FIGS / "target_distribution.png"), alt="Histograma y boxplots de exp_year")
    st.caption("El histograma muestra el sesgo a la derecha (mediana de 5.7 años) y cuántos CVs "
               "llegan al tope: 32 (0.8 %). Los boxplots muestran que las cinco ocupaciones tienen "
               "distribuciones parecidas, así que la ocupación sola no predice la experiencia.")
    st.dataframe(table("target_distribution.csv"), hide_index=True,
                 alt="Estadísticos de exp_year por ocupación")

    st.subheader("Calidad del OCR")
    st.image(str(FIGS / "ocr_quality.png"), alt="Distribución del CER y años leídos correctamente")
    st.caption("Izquierda: el CER por CV es bajo y parejo (media 0.76 %). Derecha: elegimos medir "
               "los **años** porque un dígito mal leído cambia la respuesta. El OCR crudo pierde "
               "años cuando omite el guion de un rango; tras repararlo, se recupera el 99.96 %.")
    st.dataframe(table("ocr_quality_by_occupation.csv").round(4), hide_index=True,
                 alt="Calidad del OCR por ocupación")
    with st.expander("Errores de palabra más frecuentes del OCR"):
        st.dataframe(table("ocr_top_word_errors.csv"), hide_index=True,
                     alt="Sustituciones de palabras más frecuentes")

    st.subheader("Un CV de ejemplo: cajas de PaddleOCR y texto extraído")
    examples = local_examples()
    if not examples:
        st.info("No hay imágenes locales. Se ven al ejecutar la app en Colab con Google Drive "
                "montado (las imágenes no se suben a GitHub).", title="Ejemplo no disponible")
    else:
        cv_id = st.selectbox("CV", examples)
        record = json.loads((OCR_DIR / f"{cv_id}.json").read_text(encoding="utf-8"))
        left, right = st.columns(2)
        with left:
            image = Image.open(RAW_IMG_DIR / f"{cv_id}.png")
            st.image(draw_boxes(image, record["lines"]), alt=f"CV {cv_id} con las cajas del OCR")
        with right:
            st.markdown(f"**{record['n_lines']} líneas**, confianza media "
                        f"{record['mean_score']:.3f}, {record['n_columns']} columna(s)")
            st.code(record["text"], language=None)
            st.caption("Texto en orden de lectura reconstruido a partir de las cajas. Antes del "
                       "modelo se eliminan el nombre y el contacto, y se reparan las fechas.")

# --- Tab 3 -------------------------------------------------------------------------------
with tab3:
    st.subheader("Comparación de modelos (test, 600 CVs)")
    st.dataframe(table("model_comparison.csv").round(3), hide_index=True,
                 alt="Tabla comparativa de modelos en test")

    st.subheader("Predicho vs real")
    model = st.segmented_control("Modelo", MODEL_ORDER, default="bert",
                                 format_func=lambda m: LABELS[m], key="scatter_model")
    if model:
        st.image(str(FIGS / f"pred_vs_true_{model}.png"), width=480,
                 alt=f"Predicho vs real de {LABELS[model]}")
    st.caption("La línea punteada es la de 45°. TF-IDF y los embeddings se aplanan arriba: "
               "subestiman a los seniors y sobrestiman a los juniors (tiran hacia el centro).")

    st.subheader("MAE por banda de experiencia y por ocupación")
    b1, b2 = st.columns(2)
    b1.image(str(FIGS / "mae_by_band.png"), alt="MAE por banda de experiencia")
    b2.image(str(FIGS / "mae_by_occupation.png"), alt="MAE por ocupación")
    st.image(str(FIGS / "mean_error_by_band.png"), alt="Error medio por banda", width=760)

    st.subheader("Matriz de confusión por bandas")
    cm_model = st.segmented_control("Modelo", MODEL_ORDER, default="bert",
                                    format_func=lambda m: LABELS[m], key="cm_model")
    if cm_model:
        st.image(str(FIGS / f"confusion_{cm_model}.png"), width=380,
                 alt=f"Matriz de confusión de {LABELS[cm_model]}")

    st.subheader("¿Cuánto mejora BERT? (bootstrap pareado, 2,000 remuestreos)")
    st.image(str(FIGS / "bert_gain_ci.png"), alt="Reducción de MAE de BERT con IC 95 %", width=760)
    st.dataframe(table("bert_gain_bootstrap.csv").round(3), hide_index=True,
                 alt="Ganancia de BERT por baseline")
    st.markdown("BERT mejora significativamente a los tres baselines, sobre todo en los seniors, "
                "pero **pierde frente a la regla que suma fechas**.")

    st.subheader("Pregunta 1 · ¿Lee las fechas o solo el vocabulario?")
    st.image(str(FIGS / "q1_probe.png"), alt="Prueba de fechas", width=760)
    st.markdown("Al **borrar un trabajo**, BERT resta casi exactamente su duración, como un "
                "reclutador; TF-IDF apenas reacciona. Al **desplazar todas las fechas** la "
                "experiencia no cambia, pero BERT suma +5.4 años: aprendió que 2023 es \"hoy\".")

    st.subheader("Pregunta 2 · ¿Cuánto cuesta el ruido del OCR?")
    st.dataframe(table("q2_ocr_vs_reference.csv").round(3), hide_index=True,
                 alt="MAE con texto OCR vs texto de referencia")

    st.subheader("Pregunta 3 · Sensibilidad metodológica")
    st.dataframe(table("q3_sensitivity.csv").round(3), hide_index=True,
                 alt="MAE según limpieza del OCR, dígitos en TF-IDF y pérdida de BERT")
    st.caption("La limpieza ayuda a TF-IDF y a los embeddings; quitar los dígitos casi duplica el "
               "error de TF-IDF. Las diferencias entre variantes de BERT (≤ 0.04) están dentro "
               "del ruido entre corridas.")

    st.subheader("Pregunta 4 · Chequeo de equidad de BERT")
    st.image(str(FIGS / "fairness_bert.png"), alt="Error medio de BERT por sexo y raza")
    f1, f2 = st.columns(2)
    f1.dataframe(table("q4_fairness_sex.csv").round(3), hide_index=True, alt="Equidad por sexo")
    f2.dataframe(table("q4_fairness_race.csv").round(3), hide_index=True, alt="Equidad por raza")
    st.caption("BERT sobrestima levemente a todos los grupos (+0.04 a +0.09 años). Los intervalos "
               "se solapan y el grupo Black tiene solo 29 CVs: no hay evidencia de un sesgo "
               "sistemático, y la magnitud es de semanas.")

# --- Tab 4 -------------------------------------------------------------------------------
with tab4:
    st.subheader("Estima la experiencia de un CV")
    source = st.segmented_control("Fuente", ["Subir un CV", "CV de ejemplo"],
                                  default="Subir un CV")
    image_bytes = None
    if source == "Subir un CV":
        uploaded = st.file_uploader("Imagen (PNG/JPG) o PDF; de un PDF se usa la primera página",
                                    type=["png", "jpg", "jpeg", "pdf"])
        if uploaded:
            image_bytes = to_png_bytes(uploaded)
    elif source == "CV de ejemplo":
        examples = local_examples()
        if examples:
            choice = st.selectbox("CV de ejemplo", examples, key="tab4_example")
            image_bytes = (RAW_IMG_DIR / f"{choice}.png").read_bytes()
        else:
            st.info("No hay imágenes locales disponibles.", title="Sin ejemplos")

    if image_bytes:
        try:
            record = run_ocr(image_bytes)
        except RuntimeError as e:
            st.error(f"PaddleOCR falló:\n\n```\n{e}\n```", title="Error de OCR")
            record = None
        if record:
            text = prepare_text(record["text"])
            left, right = st.columns(2)
            left.image(draw_boxes(Image.open(io.BytesIO(image_bytes)), record["lines"]),
                       alt="CV subido con las cajas del OCR")
            with right:
                st.markdown(f"**Texto extraído** ({record['n_lines']} líneas, confianza media "
                            f"{record['mean_score']:.3f}, OCR en {record['seconds']:.1f} s)")
                st.code(record["text"], language=None)
                with st.expander("Texto limpio que reciben los modelos"):
                    st.code(text, language=None)
            st.markdown("**Predicción de cada modelo**")
            show_predictions(text)

    st.divider()
    st.subheader("Explora los CVs de test")
    df = test_predictions()
    occs = st.pills("Ocupación", list(OCC_LABELS), selection_mode="multi",
                    default=list(OCC_LABELS), format_func=lambda o: OCC_LABELS[o])
    c1, c2 = st.columns(2)
    years = c1.slider("Años reales", 0.0, float(EXP_CAP), (0.0, float(EXP_CAP)), step=0.5)
    min_err = c2.slider("Error absoluto mínimo de BERT (años)", 0.0, 1.0, 0.0, step=0.05)
    view = df[df["occupation"].isin(occs or []) & df["y"].between(*years)
              & (df["bert_error"].abs() >= min_err)]
    st.caption(f"{len(view)} de {len(df)} CVs de test")
    shown = view[["cv_id", "occupation", "y"] + MODEL_ORDER + ["bert_error"]].round(2)
    selection = st.dataframe(
        shown, hide_index=True, on_select="rerun", selection_mode="single-row",
        column_config={"y": "real", **{m: LABELS[m] for m in MODEL_ORDER},
                       "bert_error": "error BERT"},
        alt="CVs de test con las predicciones de cada modelo")
    if selection.selection.rows:
        row = view.iloc[selection.selection.rows[0]]
        st.markdown(f"**{row['cv_id']}** · real {row['y']:.2f} años · BERT {row['bert']:.2f} · "
                    f"reglas {row['rules']:.2f} · TF-IDF {row['tfidf']:.2f}")
        st.code(row["text_clean"], language=None)

    st.divider()
    st.subheader("Ejemplos de buenas y malas predicciones")
    # Ejemplo bueno: ≥ 4 trabajos, lejos del tope (ahí el recorte a 20 "ayuda") y donde
    # TF-IDF falla por ≥ 1 año, para mostrar lo que BERT aporta. El mejor de BERT entre ellos.
    candidates = df[df["text_clean"].map(lambda t: len(periods(t)) >= 4)
                    & (df["y"] < EXP_CAP - 1) & ((df["tfidf"] - df["y"]).abs() >= 1)]
    good = candidates.loc[candidates["bert_error"].abs().idxmin()]
    bad = df.loc[df["bert_error"].abs().idxmax()]
    g, b = st.columns(2)
    with g.container(border=True):
        st.markdown(f":green-badge[:material/check: Buena] **{good['cv_id']}**")
        st.markdown(f"Real **{good['y']:.2f}** años · BERT **{good['bert']:.2f}** · "
                    f"TF-IDF {good['tfidf']:.2f}")
        st.code(good["text_clean"], language=None)
        st.caption(f"{len(periods(good['text_clean']))} trabajos: BERT combina los periodos y "
                   f"queda a {abs(good['bert_error']):.2f} años. TF-IDF, que solo ve años sueltos "
                   f"y vocabulario, se equivoca en {abs(good['tfidf'] - good['y']):.1f} años.")
    with b.container(border=True):
        st.markdown(f":red-badge[:material/close: Mala] **{bad['cv_id']}**")
        st.markdown(f"Real **{bad['y']:.2f}** años · BERT **{bad['bert']:.2f}** · "
                    f"reglas {bad['rules']:.2f}")
        st.code(bad["text_clean"], language=None)
        st.caption("Hay un hueco de más de un año entre dos trabajos. BERT cuenta parte del hueco "
                   "como experiencia, porque aprendió \"2023 − año de inicio\"; la regla, que "
                   "suma solo los periodos, acierta.")
