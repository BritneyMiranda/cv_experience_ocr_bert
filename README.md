# cv_experience_ocr_bert

Sistema de document AI de punta a punta: lee la imagen de un CV, extrae el texto con
**PaddleOCR** y predice los **años de experiencia laboral (0–20)** del candidato con un
modelo **BERT** ajustado como regresor. Lo comparamos con tres baselines (mediana,
TF-IDF + Ridge y sentence embeddings + Ridge).

> 🚧 Documento en construcción: cada sección se completa al cerrar su fase.

## 1. ¿Qué hace el proyecto y para quién?
_Pendiente._

## 2. Muestreo de CVs y distribución de la experiencia
_Pendiente (Fase 1)._

## 3. PaddleOCR: modelo, configuración y orden de lectura
_Pendiente (Fase 2)._

## 4. Calidad del OCR (CER, WER, años leídos correctamente)
_Pendiente (Fase 3)._

## 5. Limpieza del texto y enmascaramiento de respuestas directas
_Pendiente (Fase 3)._

## 6. Partición train / validation / test
_Pendiente (Fase 4)._

## 7. Baselines
_Pendiente (Fase 4)._

## 8. BERT como regresor
_Pendiente (Fase 5)._

## 9. Métricas finales y ganancia de BERT
_Pendiente (Fase 6)._

## 10. Preguntas analíticas
### Pregunta 1 — ¿Qué aprende BERT que los baselines no pueden?
### Pregunta 2 — ¿Cuánto cuesta el ruido del OCR?
### Pregunta 3 — Sensibilidad metodológica
### Pregunta 4 — Valor de negocio, equidad y ética

## 11. Limitaciones y aplicaciones de negocio
_Pendiente._

## 12. Cómo reproducir el proyecto
El pipeline pesado (descarga, OCR y BERT) se ejecuta en **Google Colab con GPU** usando
[notebooks/colab_pipeline.ipynb](notebooks/colab_pipeline.ipynb). Las imágenes y los pesos
de los modelos no se suben a GitHub; se regeneran con los scripts de `src/`.

1. Crear una cuenta en Hugging Face, abrir la página del dataset
   [sumin-yu/PopResume](https://huggingface.co/datasets/sumin-yu/PopResume) y aceptar
   sus condiciones de acceso.
2. Crear un token de acceso (lectura) y guardarlo en Colab → 🔑 *Secrets* con el nombre
   `HF_TOKEN`.
3. Abrir el notebook en Colab, elegir *Runtime → T4 GPU* y ejecutar las celdas en orden.

_Los pasos detallados se completan en cada fase._

## 13. Cómo ejecutar la app de Streamlit
_Pendiente (Fase 7)._
