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
**Versión y configuración** ([src/ocr_extraction.py](src/ocr_extraction.py)):
- PaddlePaddle 3.2.0 (GPU, CUDA 12.6) y PaddleOCR 3.2.0, en `lang="en"`.
- Se desactivaron la clasificación de orientación del documento, el *unwarping* y la
  orientación de líneas, porque las imágenes son escaneos limpios y derechos.
- PaddleOCR se ejecuta en un entorno aislado con Python 3.12 dentro de Colab, para no
  mezclar sus librerías CUDA con las de PyTorch.

**Elección del modelo.** El detector es `PP-OCRv5_server_det`. Para el reconocimiento
comparamos dos modelos sobre los mismos 40 CVs al azar:

| Reconocimiento | CER | WER | Seg/CV | Confianza |
|---|---:|---:|---:|---:|
| **`en_PP-OCRv5_mobile_rec`** (inglés) | **1.01 %** | **9.2 %** | **0.67** | **0.989** |
| `PP-OCRv5_server_rec` (multilingüe) | 1.05 % | 11.6 % | 0.89 | 0.982 |

Elegimos **`en_PP-OCRv5_mobile_rec`**. El CER es igual en los dos, pero el móvil tiene menos
errores de palabra (es un modelo específico para inglés) y es un 25 % más rápido. Su único
fallo sistemático es que a veces omite el guion de los rangos de fechas; lo reparamos en la
limpieza, ver la sección 5.

**Ejecución completa** ([output/metrics/ocr_summary.json](output/metrics/ocr_summary.json)):

| Métrica | Valor |
|---|---|
| Hardware | GPU Tesla T4 (Google Colab) |
| CVs procesados / fallidos | 4,000 / **0** |
| Tiempo de OCR por CV | **0.71 s** (≈ 0.95 s contando la lectura de la imagen y la escritura en Drive) |
| Tiempo total | 47 min |
| Confianza media | **0.989** (ninguna línea por debajo de 0.8) |
| Líneas por CV | 14.4 |

Cada CV se guarda en `data/ocr/<cv_id>.json` con el texto, la confianza y la caja
`[x1, y1, x2, y2]` de cada línea. El OCR se ejecuta una sola vez y después se reutiliza.

**Orden de lectura** (`reading_order`):
1. **Filas.** Las cajas se ordenan de arriba hacia abajo. Las que están a la misma altura
   (centros a menos de ½ altura de línea) se unen en una fila de izquierda a derecha, así un
   cargo no queda separado de unas fechas alineadas a la derecha. Un salto vertical grande
   abre un bloque nuevo.
2. **Dos columnas.** Se busca un "pasillo" vertical vacío en el 20–80 % central del ancho,
   de al menos 2 % del ancho. Para que cuente como pasillo, a cada lado tiene que quedar al
   menos el 15 % de las líneas, y ningún lado puede estar formado sobre todo por fechas: las
   fechas alineadas a la derecha dejan un hueco parecido, pero no son una columna. Las líneas
   que cruzan el pasillo, como un nombre centrado, actúan como separadores. Entre dos
   separadores se lee primero la columna izquierda completa y después la derecha.
3. **Pruebas.** La lógica se probó con un CV sintético de dos columnas y con uno de una
   columna con fechas a la derecha. En PopResume, **los 4,000 CVs usan una plantilla de una
   sola columna**: nombre, contacto, *Work History* y *Education*. Ninguno se detectó como de
   dos columnas, así que el orden de lectura no introduce errores en esta muestra.

## 4. Calidad del OCR (CER, WER, años leídos correctamente)
Calculado con [src/ocr_quality.py](src/ocr_quality.py) contra el texto de referencia del
dataset, que se usa **solo** para medir.

Ambos textos se normalizan igual: NFKC, minúsculas, todos los guiones como `-`, sin viñetas
ni `|`, y espacios colapsados. La versión "cruda" solo colapsa espacios.

| Ocupación | CER | WER | CER crudo | WER crudo | Años OK (crudo) | Años OK (reparado) | Rangos OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| Accountants & auditors | 0.74 % | 3.4 % | 0.99 % | 9.4 % | 96.9 % | 99.96 % | 100 % |
| Construction laborers | 0.60 % | 2.7 % | 0.91 % | 8.5 % | 97.4 % | 100 % | 100 % |
| Teachers | 0.69 % | 3.1 % | 0.86 % | 7.9 % | 98.1 % | 99.94 % | 98.1 % |
| Registered nurses | 0.87 % | 4.2 % | 1.04 % | 9.4 % | 88.7 % | 100 % | 100 % |
| Software developers | 0.92 % | 4.4 % | 1.13 % | 10.3 % | 93.3 % | 99.94 % | 100 % |
| **Total** | **0.76 %** | **3.6 %** | 0.99 % | 9.1 % | **94.5 %** | **99.96 %** | **99.7 %** |

La mediana del CER es 0.74 % y el percentil 95, 1.16 %: la calidad es muy pareja entre CVs.

**Dónde falla el OCR** ([output/tables/ocr_top_word_errors.csv](output/tables/ocr_top_word_errors.csv)):
1. **Omite el guion de los rangos de fechas.** `2017.07–2020.04` se lee como
   `2017.072020.04` en **1,337 rangos de 1,111 CVs (28 %)**. Los dígitos son correctos, pero el
   año final queda pegado al anterior: sin reparar, solo se recupera el 94.5 % de los años. La
   limpieza lo repara de forma determinista y el porcentaje sube a 99.96 %.
2. **Confunde `I` con `l` en siglas:** `API→APl`, `UI→Ul`, `CI/CD→Cl/CD`. Afecta sobre todo a
   *software developers*, que es la ocupación con el CER más alto.
3. **Corta el último carácter en las líneas más largas:** `2023.01→2023.0`, en 32 CVs de
   *teachers*, cuyo nombre de ocupación y de escuela forman la línea más larga. Es la causa de
   su 98.1 % en rangos. El año sí se conserva; se pierde un dígito del mes.
4. **La confianza no predice el error:** la correlación entre la confianza media y el CER es
   **0.04**.

En total, **3,990 de los 4,000 CVs (99.75 %) tienen todos sus años correctos** después de
la reparación.

## 5. Limpieza del texto y enmascaramiento de respuestas directas
[src/preprocessing.py](src/preprocessing.py) genera `data/processed/cvs.csv`.

**Limpieza:**
- Normalización NFKC.
- Guiones `– — −` → `-`.
- **Reparación de rangos de fechas:** todos quedan como `AAAA.MM - AAAA.MM`; 1,337 rangos no
  tenían guion.
- Viñetas `•` → `- `, y se quita el espacio que deja una viñeta perdida por el OCR.
- Se unen las palabras cortadas con guion al final de línea y las líneas que continúan una
  frase.

**Líneas de baja confianza.** No se eliminan en el texto principal. Ninguna línea tiene
confianza menor que 0.8, y la confianza no se correlaciona con el error (r = 0.04): con un
umbral se borraría texto correcto sin ganar nada. Como variante para la Pregunta 3,
`text_clean_conf` elimina las líneas con confianza < 0.9: son 31 líneas en 31 CVs.

**Datos personales eliminados:**
- Email (4,000), teléfono (3,985) y ubicación `XX, USA` (4,000), que además es el estado,
  una variable prohibida.
- El nombre, que se busca solo en el encabezado: 8,000 apariciones.

Se verificó que en ningún CV queda texto antes de *Work History*, ni rastros de contacto
(`+`, `@`, `USA`). En los 15 CVs donde no se contó un teléfono, el OCR había pegado el teléfono
al email sin espacios (`...-9216|nombre@...`), y la regex del email eliminó ambos juntos.

**Respuestas directas.** Se buscan con regex:
- experiencia total ("12 years of experience", "Experience: 7 years");
- edad ("Age: 39", "34 years old");
- fecha de nacimiento ("Born in 1990", "DOB").

Las fechas de cada trabajo y los años de graduación **no** se tocan. Las regex se probaron
con frases de ejemplo y se leyeron **50 CVs al azar**
([output/tables/review_sample_50.txt](output/tables/review_sample_50.txt)). **Ningún CV contiene
estas frases: 0 CVs enmascarados.** La plantilla de PopResume no tiene sección de resumen.

**Hallazgo de la lectura manual.** `exp_year` coincide con la **suma de la duración de los
trabajos**, sin contar los periodos entre trabajos. Por ejemplo:
- 2020.01–2023.01 + 2015.03–2018.03 = 6.0 años, y `exp_year` = 6.00;
- un CV con un hueco entre 2009 y 2012 suma 4 × 3.5 = 14.04 años.

Todos los trabajos actuales terminan en 2023.01, así que la fecha de referencia del dataset
es enero de 2023. Esto hace que un modelo de reglas que sume los periodos sea un competidor
muy fuerte para BERT (ver la sección 7).

**Versiones del texto en `cvs.csv`:**

| Columna | Contenido | Uso |
|---|---|---|
| `text_raw` | OCR en orden de lectura, sin PII y con enmascarado | Pregunta 3 |
| `text_clean` | + limpieza | **Entrada de todos los modelos** |
| `text_clean_conf` | + sin líneas con confianza < 0.9 | Pregunta 3 |
| `text_ref` | Texto de referencia con la misma limpieza | Solo Pregunta 2 (nunca para entrenar) |

`sex` y `race` se conservan únicamente para el chequeo de equidad de la Pregunta 4. `age`,
`state`, `region` y los nombres no pasan a `cvs.csv`.

Un CV limpio tiene en promedio **516 caracteres y 65 palabras**, unos ~130 tokens de BERT,
así que cabe completo en 512 tokens.

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
