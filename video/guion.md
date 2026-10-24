# Guion del video (≤ 5 min)

Formato *pitch*: se muestra la app y las figuras, nunca el código. Unas 680 palabras habladas,
alrededor de 4 min 40 s a ritmo tranquilo. **[Pantalla]** indica qué mostrar.

---

## 1. El problema (0:00 – 0:35)
**[Pantalla]** App, pestaña *Problema y metodología*, con las 4 métricas arriba.

> Las agencias de empleo, los portales y los servicios públicos reciben miles de CVs
> escaneados o en PDF. Un dato básico para ordenarlos es cuántos años de experiencia tiene
> cada persona. El problema es que ese número casi nunca está escrito: un reclutador lo
> calcula mirando las fechas de cada trabajo. Nuestro sistema lee la imagen del CV y estima
> esa cifra, de 0 a 20 años. Trabajamos con 4,000 CVs sintéticos de PopResume, de cinco
> ocupaciones, generados con estadísticas de EE. UU.: ninguno es de una persona real.

## 2. De la imagen al texto (0:35 – 1:20)
**[Pantalla]** Pestaña *OCR y exploración*: el CV de ejemplo con las cajas azules y el
texto al lado. Luego la figura de calidad del OCR.

> Un CV escaneado es solo una imagen, así que primero hay que leerlo. Usamos PaddleOCR:
> detecta cada línea, la transcribe y nos da su posición y su confianza. Con esas cajas
> reconstruimos el orden de lectura, para que un cargo no quede separado de sus fechas.
> El OCR es muy bueno: se equivoca en menos del 1 % de los caracteres. Pero tiene una falla
> que importa: en el 28 % de los CVs se "come" el guion entre dos fechas, y "2017.07 –
> 2020.04" queda como un solo número. Sin arreglarlo, solo el 94 % de los años quedan bien
> leídos; reparándolo, el 99.96 %. Además borramos nombre, email, teléfono y estado, que no
> deben influir en el modelo.

## 3. Por qué es difícil (1:20 – 1:55)
**[Pantalla]** Pestaña *Estimador interactivo* con un CV de ejemplo de enfermería con cuatro
trabajos (por ejemplo `registered_nurses_03093`).

> Miren este CV de enfermería. No dice "12 años de experiencia". Tiene cuatro trabajos:
> 2010 a 2014, 2014 a 2017, 2017 a 2020 y 2020 a 2023. Para llegar a 12 años hay que
> encontrar cada periodo, restar fin menos inicio y sumarlos, y si hay un hueco entre dos
> trabajos, no contarlo. Eso es lo que el modelo tiene que aprender a partir del texto.

## 4. Cuatro formas de adivinar (1:55 – 2:40)
**[Pantalla]** Pestaña 1, la tabla de modelos.

> Comparamos cuatro enfoques sobre exactamente los mismos datos.
> Primero, la mediana: responder siempre 5.6 años, sin leer nada. Es el piso.
> Segundo, TF-IDF: cuenta palabras. Aprende que "2008" suele significar más experiencia y
> "2021" menos, pero no puede restar fechas.
> Tercero, embeddings de frases: resumen de qué trata el CV, pero casi ignoran los números.
> Cuarto, BERT ajustado: lee el CV completo y aprende a combinar las fechas.
> Y como prueba honesta, agregamos una regla de 40 líneas que solo extrae las fechas y suma
> los periodos.

## 5. Resultados (2:40 – 3:45)
**[Pantalla]** Pestaña *Resultados*: tabla comparativa, luego "Predicho vs real" (BERT y luego
TF-IDF), luego los dos ejemplos, bueno y malo, de la pestaña 4.

> En el test, la mediana se equivoca 3.6 años en promedio. TF-IDF baja a 1 año, y los
> embeddings a 1.3. BERT baja a 0.16 años, unos dos meses, con el 100 % de los CVs dentro de
> ±2 años. TF-IDF llega al 90 %.
> En el gráfico se ve la diferencia: los puntos de BERT siguen la diagonal, mientras que
> TF-IDF se aplana arriba y subestima a los seniors en más de un año.
> Un ejemplo bueno: este desarrollador tiene cuatro trabajos y 13.7 años de experiencia.
> BERT predice exactamente 13.7; TF-IDF se queda en 12.2.
> Un ejemplo malo: este enfermero tiene un hueco de casi dos años entre trabajos. BERT
> predice 17.3 años, cuando son 16.4. Cuenta parte del hueco como experiencia. La regla
> acierta.

## 6. ¿Vale la pena BERT? ¿Es justo? (3:45 – 4:45)
**[Pantalla]** Figura de la ganancia con intervalos de confianza, luego la prueba de fechas,
luego la figura de equidad.

> ¿Vale la pena BERT? Frente a los baselines sí: ahorra 0.8 años de error frente a TF-IDF, y
> el intervalo de confianza no incluye el cero. Pero la regla que suma fechas es siete veces
> más precisa que BERT, no necesita GPU y se puede explicar.
> Además hicimos una prueba: si borramos un trabajo, BERT resta su duración, como un
> reclutador. Pero si movemos todas las fechas cinco años hacia atrás, sin cambiar las
> duraciones, BERT suma cinco años. Aprendió que 2023 es "hoy", porque en estos datos todos
> los trabajos actuales terminan ahí. Con CVs de otro año fallaría.
> En equidad, BERT sobrestima a todos los grupos por igual, entre dos y cinco semanas, sin
> diferencias claras por sexo ni por raza.
> Nuestra recomendación: para documentos como estos, OCR más una regla de fechas, y BERT solo
> si los formatos son variados. Y en cualquier caso, esto sirve para prellenar un perfil que
> una persona revisa. Nunca debe decidir a quién se entrevista o se contrata.

---

## Consejos para grabar
- Ten la app abierta en Colab antes de empezar y precarga la pestaña 4 con un CV de ejemplo:
  la primera predicción tarda unos segundos.
- Si te pasas de 5 minutos, recorta primero la sección 4: la tabla de modelos ya se explica
  sola.
- Al terminar, pega el enlace del video en `video/link.txt` (YouTube "no listado" o Google
  Drive con permiso de lectura).
