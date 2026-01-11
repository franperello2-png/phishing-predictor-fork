from flask import Flask, render_template, request, send_file, redirect, url_for
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
import io
from gridfs import GridFS
import os
from urllib.parse import urlparse, parse_qs, quote_plus
import random
import cohere
import onnxruntime as ort
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime



load_dotenv()
user = os.getenv("MONGO_USERNAME") #en el archivo .env poned vuestros datos de usuario
pw = os.getenv("PASSWORD")
cohere_api_key = os.getenv("COHERE_API_KEY")

co = cohere.ClientV2(api_key=cohere_api_key)

app = Flask(__name__)


def analisis_img(img_url):
    """
    Cohere analiza la imagen ofrecida del usuario y devuelve un JSON con la predicción,
    si la imagen introducida del usuario es válida o no y un porcentaje de riesgo de phishing
    """
    prompt = """
    Eres un sistema de detección de phishing a partir de imágenes de mensajes.

    Tu tarea es:
    1. Determinar si la imagen proporcionada contiene un mensaje legible (correo electrónico, SMS, notificación o chat).
    2. Si la imagen es un mensaje:
    a. Determinar si es un intento de phishing o no.
    b. Devolver un porcentaje que indique la probabilidad de que sea phishing.
    3. Si la imagen NO es un mensaje, indícalo claramente.

    Devuelve SIEMPRE un JSON válido con las siguientes claves EXACTAS:

    - "es_mensaje": boolean
    - "es_phishing": boolean o false si no aplica
    - "probabilidad_phishing": número entre 0 y 100 o 0 si no aplica
    - "explicacion": string con una explicación clara y comprensible

    Ejemplos:

    Imagen NO es un mensaje:
    {
    "es_mensaje": false,
    "es_phishing": false,
    "probabilidad_phishing": 0,
    "explicacion": "La imagen no contiene un mensaje legible, por lo que no se puede evaluar phishing."
    }

    Imagen es mensaje y NO es phishing:
    {
    "es_mensaje": true,
    "es_phishing": false,
    "probabilidad_phishing": 8,
    "explicacion": "El mensaje no presenta señales típicas de phishing como urgencia o solicitudes de datos sensibles."
    }

    Imagen es mensaje y SÍ es phishing:
    {
    "es_mensaje": true,
    "es_phishing": true,
    "probabilidad_phishing": 91,
    "explicacion": "El mensaje utiliza lenguaje urgente y suplanta a una entidad legítima solicitando una acción inmediata."
    }

    Instrucciones:
    - Devuelve SOLO JSON válido.
    - No añadas texto fuera del JSON.
    - No inventes datos.
    """
    try:
        response = co.chat(
            model="command-a-vision-07-2025",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_url, "detail": "auto"}}
                    ]
                }
            ]
        )
        # Cohere devuelve un objeto con response.message.content[0].text
        return json.loads(response.message.content[0].text)

    except Exception as e:
        print(f"Error procesando imagen en Cohere: {e}")
        return None  # devuelve None si falla
    
def get_explanation(text, prediction): #texto del usuario y prediccion del modelo 
    """
    Aquí la IA de Cohere tiene que averiguar y explicar el porqué el modelo ha dado los resultados que ha dado.
    """

    label = "phishing" if prediction == 1 else "no phishing"

    prompt = f"""
    Eres un experto en ciberseguridad.
    Un modelo automático ha analizado el siguiente texto y ha concluido que ES {label.upper()}.
    Texto analizado:
    \"\"\"{text}\"\"\"

    Explica de forma clara y sencilla por qué este texto puede considerarse {label}.
    Menciona señales típicas de phishing si existen, como urgencia, suplantación de identidad,
    enlaces sospechosos o solicitudes de información sensible.
    No inventes datos.
    """
    try:
        response = co.generate(
            model="command",
            prompt=prompt,
            max_tokens=200,
            temperature=0.3
        )

        story = response.message.content[0].text
        if story:
            return story.strip()

        # Si cohere falla y no devuelve una historia
        raise RuntimeError("Cohere returned empty story")

    except Exception as e:
        #Envía historia en caso de fallo 
        print(f"Warning: Cohere failed to generate story from text: {e}")
        if label == "phishing":
            fallbacks = [
                "Este texto presenta varias características comunes en mensajes de phishing."
                "Se ha detectado un tono de urgencia y una posible solicitud de acción inmediata, lo cual es una técnica habitual utilizada para engañar al usuario. Además, el contenido puede estar relacionado con la suplantación de identidad de una entidad legítima."
                "Por seguridad, se recomienda no hacer clic en enlaces, no responder al mensaje y verificar la información directamente con la entidad oficial."
            ]
            return random.choice(fallbacks)
        else:
            fallbacks = [
                "El texto analizado no muestra señales claras asociadas a phishing.",
                "No se han detectado solicitudes de información sensible ni lenguaje alarmista. El contenido parece informativo y coherente, sin patrones típicos de engaño.",
                "Aun así, se recomienda mantener precaución y verificar siempre el origen del mensaje."
            ]
            return random.choice(fallbacks)
    
def get_db():
    """
    Crear y comprobar la conexión a MongoDB y devolver la base de datos que se va a usar
    """
    username = quote_plus(user)
    password = quote_plus(pw)

    uri = f"mongodb+srv://{username}:{password}@cluster0.naqjxci.mongodb.net/?appName=Cluster0"
    
    client = MongoClient(uri, server_api=ServerApi('1'))

    try:

        client.admin.command('ping')

        print("Conexión con MongoDB exitosa")

        return client['DB_Phishing']
    
    except Exception as e:

        print(f"Error conectando a MongoDB: {e}")

        raise

db = None

@app.before_request
def connect_db():
    """
    Asegura que exista una conexión antes de cada request
    """
    global db
    if db is None:
        db = get_db()

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Página principal
    Esencial: botones para dirigir al usuario a las otras rutas (abajo)
    """
    return render_template('home.html')

@app.route('/history', methods=['GET', 'POST'])
def history():
    """
    Página que enseña los datos historicos (todos lo que tenemos en la bd)
    """
    return None

@app.route('/stats', methods=['GET', 'POST'])
def stats():
    """
    Enseña gráficas (debatir que gráficas mostrar y su diseño)
    """
    return None

@app.route('/report', methods=['GET', 'POST'])
def report():
    """
    Formulario que tiene que rellenar el usuario para poder reportar el intento de phishing.
    Cada vez que el usuario envía algo tiene que subirse a la db
    """
    link_encontrado = None
    collection_links = db["report_reliable_links"]
    collection_antivirus = db["best antivirus"]
    report_links = collection_links.find_one({}, {"_id": 0})
    antivirus_list = list(collection_antivirus.find({}, {"_id": 0}))

    if request.method == "POST":
        tipo = request.form.get("tipo")
        pais = request.form.get("pais", "Global")
        link_encontrado = report_links.get(tipo, {}).get(pais, report_links.get(tipo, {}).get("Global"))

    return render_template(
        "report.html",
        report_links=report_links,
        link_encontrado=link_encontrado,
        antivirus_list=antivirus_list
    )

@app.route('/advising', methods=['GET', 'POST'])
def advising():
    """
    Contiene las FAQ en relación al phishing, también contiene links (o vídeos insertados) explicando dudas y
    definiciones simples para ayudar a la persona a detectar posibles scams
    """
    return render_template("advising.html")

   
# @app.route("/serve_image/<file_id>")
# def serve_image(file_id):
#     fs = GridFS(db)
#     file = fs.get(ObjectId(file_id))
#     return send_file(io.BytesIO(file.read()), mimetype=file.contentType)

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    """
    Da las opciones de predecir por imagen (o URL de la imagen), texto o URL de página web dudosa

    A partir de una imagen o URL a la imagen, Cohere detecta si es un mensaje. En caso de que lo sea
    identifica si se trata de phishing o no y su porcentaje de phishing, por lo contrario envía un mensaje
    al usuario explicando que la imagen no es válida para la predicción
    """
    resultado_url = None
    resultado_img = None

    # Verificar si envían URL
    img_url = request.form.get("img_url")
    if img_url:
        res = analisis_img(img_url)
        if res is None:
            resultado_url = {"error": "Error al procesar la imagen, pruebe con otra."}
        else:
            resultado_url = res

    # Verificar si envían archivo (desplegar en render para que cohere pueda leer la imagen bytes)
    img_file = request.files.get("img_file")
    if img_file:
        # Guardar temporalmente para obtener URL accesible, o pasar bytes a Cohere
        # Para simplificar asumimos que subes a un endpoint público o lo pasas como URL de prueba
        # Aquí solo usamos una URL de ejemplo
        res = analisis_img("https://ejemplo.com/imagen_subida.png")
        if res is None:
            resultado_img = {"error": "Error al procesar la imagen subida."}
        else:
            resultado_img = res

    return render_template(
        "predictions.html",
        resultado_url=resultado_url,
        resultado_img=resultado_img
    )
@app.route("/minigame/image/<image_id>")
def minigame_image(image_id):
    fs = GridFS(db)
    file = fs.get(ObjectId(image_id))
    return send_file(io.BytesIO(file.read()), mimetype=file.contentType)

@app.route('/minigame', methods=['GET', 'POST'])
def minigame():
    """
    Minijuego, el usuario tiene que adivinar si un mensaje o screenshot de una página es una scam o no
    """
    images = list(db["minigame"].find())
    if not images:
        return "No hay imágenes en la base de datos."

    result = None

    if request.method == "POST":
        image_id = request.form.get("image_id")
        image = db["minigame"].find_one({"_id": ObjectId(image_id)})

        user_answer = request.form.get("answer")
        user_answer_bool = True if user_answer == "true" else False

        result = {
            "correct": user_answer_bool == image["is_phishing"],
            "correct_answer": image["is_phishing"]
        }

    else:
        image = random.choice(images)
        image_id = str(image["_id"])

    return render_template(
        "minigame.html",
        image_id=image_id,
        result=result
    )


if __name__ == "__main__":
    app.run(debug = True, host = "localhost", port  = 5000)