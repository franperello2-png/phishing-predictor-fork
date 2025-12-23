from flask import Flask, render_template, request

app = Flask(__name__)

def get_db():
    """
    Crear y comprobar la conexión a MongoDB y devolver la base de datos que se va a usar
    """
    return None

@app.before_request
def connect_db():
    """
    Asegura que exista una conexión antes de cada request
    """
    return None

@app.route('/', methods=['GET'])
def home():
    """
    Página principal
    Esencial: botones para dirigir al usuario a las otras rutas (abajo)
    """
    return None

@app.route('/history', methods=['GET'])
def history():
    """
    Página que enseña los datos historicos (todos lo que tenemos en la bd)
    """
    return None

@app.route('/stats', methods=['GET'])
def stats():
    """
    Enseña gráficas (debatir que gráficas mostrar y su diseño)
    """
    return None

@app.route('/report', methods=['GET'])
def report():
    """
    Permite reportar una página a partir de su URL y también permite reportar una cuenta email
    """
    return None

@app.route('/advising', methods=['GET'])
def advising():
    """
    Contiene las FAQ en relación al phishing, también contiene links (o vídeos insertados) explicando dudas y
    definiciones simples para ayudar a la persona a detectar posibles scams
    """
    return None

@app.route('/predictions', methods=['GET'])
def predictions():
    """
    Predice si una URL, Imagen o texto son phishing o no. También devuelve un rango del riesgo
    """
    return None

@app.route('/minigame', methods=['GET'])
def minigame():
    """
    Minijuego, el usuario tiene que adivinar si un mensaje o screenshot de una página es una scam o no
    """
    return None