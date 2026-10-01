"""
AgroDetect - Detección de Enfermedades en Hojas de Café
Servicio Web en Streamlit + TensorFlow + API de Groq
"""

import json
import os
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from groq import Groq

# ----------------------------------------------------------------------
# Configuración general de la página
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AgroDetect - Diagnóstico de Café",
    page_icon="🌿",
    layout="wide",
)

MODEL_PATH = "modelo_cafe.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

# Diccionario para mostrar nombres "bonitos" en pantalla en vez del nombre
# técnico de la carpeta del dataset. Ajusta esto según tus clases reales.
NOMBRES_VISIBLES = {
    "healthy": "Hoja Sana",
    "rust": "Roya del Café",
    "leaf_miner": "Minador de la Hoja",
    "phoma": "Phoma / Mancha de Hierro",
    "cercospora": "Cercospora / Mancha de Hierro",
    "red_spider": "Ácaro / Araña Roja",
}


# ----------------------------------------------------------------------
# Carga de modelo y cliente de Groq (con caché para no recargar cada vez)
# ----------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    modelo = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        clases = json.load(f)
    return modelo, clases


def obtener_cliente_groq():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ----------------------------------------------------------------------
# Lógica de predicción
# ----------------------------------------------------------------------
def predecir_enfermedad(modelo, clases, imagen: Image.Image):
    img_resized = imagen.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img_resized), axis=0).astype("float32")

    predicciones = modelo.predict(arr, verbose=0)[0]
    idx = int(np.argmax(predicciones))
    clase = clases[idx]
    confianza = float(predicciones[idx]) * 100

    return clase, confianza, predicciones


def generar_recomendacion_groq(client: Groq, clase: str, confianza: float):
    nombre_visible = NOMBRES_VISIBLES.get(clase, clase)

    prompt = f"""Eres un técnico agrónomo experto en el cultivo de café, similar a un asesor de IHCAFE
(Instituto Hondureño del Café). Se detectó la siguiente condición en una hoja de café mediante un
modelo de visión artificial:

Diagnóstico: {nombre_visible}
Confianza del modelo: {confianza:.1f}%

Genera una orientación técnica breve y clara en español, en formato Markdown, con estas 4 secciones
numeradas (usa encabezados en negrita):

1. **Diferenciación a simple vista** - cómo reconocer esta condición y con qué se puede confundir.
2. **Manejo agronómico preventivo y correctivo** - acciones concretas de fertilización, sombra,
   fungicidas/control según aplique.
3. **Monitoreo y seguimiento** - cada cuánto revisar y qué señales indican mejora o empeoramiento.
4. **Buenas prácticas y registro** - qué documentar para el manejo integral del cultivo.

Si el diagnóstico es "Hoja Sana", ajusta el contenido a recomendaciones de mantenimiento preventivo
en vez de tratamiento. Sé específico pero conciso (máximo 2-3 líneas por sección)."""

    respuesta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=700,
    )
    return respuesta.choices[0].message.content


# ----------------------------------------------------------------------
# Estado de sesión (historial de diagnósticos)
# ----------------------------------------------------------------------
if "historial" not in st.session_state:
    st.session_state.historial = []

# ----------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------
col_izq, col_der = st.columns([1, 1.3], gap="large")

with col_izq:
    st.title("🌿 Captura de Imagen Foliar")
    st.caption(
        "Posicione la hoja de café bajo luz natural. El sistema detectará "
        "automáticamente signos de Roya, Cercospora o Plagas."
    )

    fuente = st.radio("Fuente de la imagen", ["Subir archivo", "Usar cámara"], horizontal=True)

    if fuente == "Subir archivo":
        archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    else:
        archivo = st.camera_input("Toma una foto de la hoja")

    if archivo is not None:
        imagen = Image.open(archivo)
        st.image(imagen, use_container_width=True)

with col_der:
    st.subheader("Último diagnóstico")

    if archivo is None:
        st.info("Sube o captura una imagen para iniciar el diagnóstico.")
    else:
        modelo, clases = cargar_modelo()
        clase, confianza, _ = predecir_enfermedad(modelo, clases, imagen)
        nombre_visible = NOMBRES_VISIBLES.get(clase, clase)

        st.markdown(f"### {nombre_visible}")
        st.metric("Confianza del modelo", f"{confianza:.1f}%")

        client = obtener_cliente_groq()

        if client is None:
            st.warning(
                "No se encontró la API key de Groq. Configúrala en `.streamlit/secrets.toml` "
                "(local) o en Settings → Secrets (Streamlit Community Cloud) como `GROQ_API_KEY`."
            )
        else:
            with st.spinner("Generando orientación técnica con IA..."):
                try:
                    recomendacion = generar_recomendacion_groq(client, clase, confianza)
                    st.markdown("---")
                    st.markdown(recomendacion)

                    st.session_state.historial.insert(
                        0,
                        {
                            "clase": nombre_visible,
                            "confianza": confianza,
                            "fecha": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        },
                    )
                except Exception as e:
                    st.error(f"No se pudo generar la recomendación con Groq: {e}")

    if st.session_state.historial:
        st.markdown("---")
        st.caption("HISTORIAL RECIENTE")
        for item in st.session_state.historial[:5]:
            c1, c2 = st.columns([3, 1])
            c1.write(f"**{item['clase']}**")
            c2.caption(item["fecha"])
