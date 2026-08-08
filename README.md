# AgroDetect - Detección de Enfermedades en Hojas de Café

Servicio Web basado en Computación en la Nube que detecta enfermedades en hojas de café
mediante un modelo de visión artificial (TensorFlow) y genera recomendaciones técnicas de
manejo preventivo mediante la API de Groq.

## Arquitectura del sistema

- **Frontend/Backend:** Streamlit (aplicación web en Python de una sola capa).
- **Modelo de IA:** Red convolucional basada en MobileNetV2 (transfer learning) entrenada en
  Google Colab sobre un dataset de hojas de café (clases: sana, roya, minador, phoma).
- **Generación de recomendaciones:** API de Groq (modelo `llama-3.3-70b-versatile`), a partir
  de la clase detectada y el porcentaje de confianza.
- **Despliegue:** Streamlit Community Cloud.

## Flujo de funcionamiento

1. El usuario sube una imagen o toma una foto de una hoja de café.
2. La imagen se redimensiona a 224x224 y se pasa al modelo TensorFlow.
3. El modelo devuelve la clase predicha y el porcentaje de confianza.
4. Con ese resultado se arma un prompt que se envía a la API de Groq.
5. Groq genera la orientación técnica (diferenciación visual, manejo agronómico, monitoreo
   y buenas prácticas), que se muestra en pantalla junto al historial de diagnósticos.

## Instalación y ejecución local

1. Clona el repositorio:
   ```bash
   git clone <url-de-tu-repositorio>
   cd agrodetect
   ```

2. Crea un entorno virtual e instala dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate      # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Coloca en la raíz del proyecto los archivos generados en Colab:
   - `modelo_cafe.keras`
   - `class_names.json`

4. Configura tu API key de Groq. Copia el archivo de ejemplo y agrega tu key real:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Edita `.streamlit/secrets.toml` y reemplaza el valor de `GROQ_API_KEY` con tu key de
   [console.groq.com](https://console.groq.com).

5. Ejecuta la aplicación:
   ```bash
   streamlit run app.py
   ```

## Despliegue en Streamlit Community Cloud

1. Sube el repositorio a GitHub (público), **sin el archivo `secrets.toml` real**.
2. Entra a [share.streamlit.io](https://share.streamlit.io) y conecta tu repositorio.
3. En **Settings → Secrets**, agrega:
   ```
   GROQ_API_KEY = "tu_api_key_real"
   ```
4. Asegúrate de que `modelo_cafe.keras` y `class_names.json` estén incluidos en el repositorio
   (o gestionados con Git LFS si el modelo pesa más de 100 MB).
5. Despliega. Streamlit instalará automáticamente lo indicado en `requirements.txt`.

## Tecnologías empleadas

- Python 3
- Streamlit
- TensorFlow / Keras
- Groq API
- Pillow (procesamiento de imágenes)
- NumPy

## Nota sobre el dataset

El dataset original del curso incluía 6 clases (roya, sana, minador, phoma, cercospora,
araña roja). Los archivos correspondientes a *cercospora* y *araña roja* resultaron ser
punteros de Git LFS corruptos (apuntadores de ~130 bytes en vez del archivo real), por lo
que el modelo actual fue entrenado únicamente con las 4 clases cuyo contenido estaba íntegro:
sana, roya, minador y phoma.
