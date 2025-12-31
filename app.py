from flask import Flask, render_template, request
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from urllib.parse import urlparse, parse_qs, quote_plus

load_dotenv()
user = os.getenv("MONGO_USERNAME") #en el archivo .env poned vuestros datos de usuario
pw = os.getenv("PASSWORD")

app = Flask(__name__)

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
    Permite reportar una página a partir de su URL y también permite reportar una cuenta email
    """
    return None

@app.route('/advising', methods=['GET', 'POST'])
def advising():
    """
    Contiene las FAQ en relación al phishing, también contiene links (o vídeos insertados) explicando dudas y
    definiciones simples para ayudar a la persona a detectar posibles scams
    """
    faqs = [
        {
            "question": "¿Qué es el phishing?",
            "answer": "El phishing es un intento de engaño para robar información personal haciéndose pasar por una entidad legítima."
        },
        {
            "question": "¿Cómo identificar un correo phishing?",
            "answer": "Revisa el remitente, enlaces sospechosos, errores ortográficos y mensajes de urgencia."
        },
        {
            "question": "¿Qué hago si he caído en un phishing?",
            "answer": "Cambia tus contraseñas inmediatamente y contacta con tu banco o proveedor."
        }
    ]

    videos = [
        {
            "title": "¿Qué es el phishing?",
            "youtube_id": "XBkzBrXlle0"
        },
        {
            "title": "Cómo evitar ataques de phishing",
            "youtube_id": "kL1zYx6RrKo"
        }
    ]

    return render_template("advising.html", faqs=faqs, videos=videos)

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    """
    Predice si una URL, Imagen o texto son phishing o no. También devuelve un rango del riesgo
    """
    return None

@app.route('/minigame', methods=['GET', 'POST'])
def minigame():
    """
    Minijuego, el usuario tiene que adivinar si un mensaje o screenshot de una página es una scam o no
    """
    return None

if __name__ == "__main__":
    app.run(debug = True, host = "localhost", port  = 5000)