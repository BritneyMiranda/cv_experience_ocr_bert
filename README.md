# cv_experience_ocr_bert

Sistema de document AI de punta a punta: lee la imagen de un CV, extrae el texto con
**PaddleOCR** y predice los **años de experiencia laboral (0–20)** del candidato con un
modelo **BERT** ajustado como regresor. Lo comparamos con tres baselines (mediana,
TF-IDF + Ridge y sentence embeddings + Ridge).

> 🚧 Documento en construcción: cada sección se completa al cerrar su fase.

## 1. ¿Qué hace el proyecto y para quién?
_Pendiente._

## 2. Muestreo de CVs y distribución de la experiencia
**Dataset:** [PopResume](https://huggingface.co/datasets/sumin-yu/PopResume) (Yu, Park & Moon,
EMNLP 2026; CC BY 4.0), CVs sintéticos generados a partir de estadísticas de población de EE. UU.

**Muestreo** ([src/download_data.py](src/download_data.py)):
- Solo imágenes **sin foto** (`has_photo == False`).
- **800 CVs por ocupación, 4,000 en total**, con semilla **42**.
- Se hace en dos pasos. Primero se lee un índice liviano de los 43 shards (todas las columnas
  menos la imagen) y se sortean los CVs con `numpy.random.default_rng(42)`. Esto da un muestreo
  aleatorio **uniforme** sobre toda la población sin foto, sin depender del tamaño de un buffer de
  mezcla como en el streaming. Después se descarga cada shard, se extraen solo las imágenes
  sorteadas y el shard se borra.
- El `id` de PopResume se repite entre ocupaciones, así que la clave del proyecto es
  `cv_id = "<job>_<id>"`. Ningún CV tenía más de una imagen sin foto.
- Los IDs sorteados están en [data/raw/sampled_ids.csv](data/raw/sampled_ids.csv). Con ese
  archivo, o con la misma semilla, la muestra se reproduce exactamente.

| Ocupación | Población sin foto | Muestra | Media | Mediana | Máx. | CVs ≥ 20 años |
|---|---:|---:|---:|---:|---:|---:|
| Accountants & auditors | 7,784 | 800 | 7.27 | 6.24 | 25.6 | 11 |
| Construction laborers | 5,910 | 800 | 5.96 | 4.60 | 22.6 | 10 |
| Elementary / middle school teachers | 16,188 | 800 | 6.80 | 5.95 | 26.7 | 6 |
| Registered nurses | 17,632 | 800 | 6.46 | 5.63 | 24.0 | 2 |
| Software developers | 13,370 | 800 | 6.33 | 5.64 | 22.4 | 3 |
| **Total** | **60,884** | **4,000** | **6.56** | **5.66** | **26.7** | **32** |

**Target:** `exp_year`, los años de experiencia, que en el dataset es un valor continuo (por
ejemplo 3.44). Se recorta a **0–20**, y 20 se lee como "20 o más".

**Revisión del tope:** solo **32 CVs (0.8 %)** llegan a 20 años o más, muy por debajo del umbral
del 15 %. Por eso se **mantiene el tope en 20** para todos los modelos.

La distribución está sesgada a la derecha: la mediana es 5.7 años y hay pocos CVs senior. Por
eso esperamos que los modelos tiendan a "tirar hacia el centro", sobrestimando a los juniors y
subestimando a los seniors (ver la Pregunta 1).

Tabla completa: [output/tables/target_distribution.csv](output/tables/target_distribution.csv).

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
