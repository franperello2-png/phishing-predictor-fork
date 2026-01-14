from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
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
import base64
from datetime import datetime, timezone




load_dotenv()
user = os.getenv("MONGO_USERNAME") #en el archivo .env poned vuestros datos de usuario
pw = os.getenv("PASSWORD")
cohere_api_key = os.getenv("COHERE_API_KEY")

co = cohere.ClientV2(api_key=cohere_api_key)

app = Flask(__name__)

print("Cargando Modelos")
try:
    #Cargamos Modelo para URLs
    sess_url = ort.InferenceSession("detector_phishing_uci.onnx")
    input_name_url = sess_url.get_inputs()[0].name
    label_name_url = sess_url.get_outputs()[0].name
    
    #Cargamos Modelo para Texto de Mensajes
    sess_text = ort.InferenceSession("detector_fraude_final.onnx")
    input_name_text = sess_text.get_inputs()[0].name
    label_name_text = sess_text.get_outputs()[0].name
    print("Modelos ONNX cargados correctamente.")
except Exception as e:
    print(f"Error al caargar alguno de los modelos: {e}")
    sess_url = None
    sess_text = None

FEATURES_ORDER = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 
    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 
    'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL', 
    'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL', 
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 
    'Statistical_report'
]

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

def analisis_texto_onnx(texto_usuario):
    """
    Analiza el texto de un mensaje usando el modelo ONNX de texto previamente entrenado.
    """
    if not sess_text:
        return {"Error":"El Modelo de texto no se ha cargado correctamente."}

    try:
        inputs = {input_name_text: np.array([[texto_usuario]], dtype=object)}
        prediccion = sess_text.run(None, inputs)
        label = prediccion[0][0]
        probs = prediccion[1][0] #Mapa de probabilidades
        
        # Interpretación (ajustar según tu dataset: 1/'spam' = fraude)
        es_fraude = (str(label).lower() in ['1', 'spam', 'phishing'])
        
        # Obtener confianza
        confianza = probs.get(label, 0) * 100

        pred_int = 1 if es_fraude else 0
        # Usamos la función original get_explanation hecha previamente para generar la explicación
        explicacion_ia = get_explanation(texto_usuario, pred_int)

        return {
            "es_phishing": es_fraude,
            "probabilidad": round(confianza, 2),
            "tipo": "Texto (SMS/Email)",
            "explicacion": explicacion_ia
        }
    except Exception as e:
        print(f"Error en ONNX Texto: {e}")
        return {"error": "Error interno al analizar el texto."}
    

def analisis_url_hibrido(url_usuario):
    """
    IA HÍBRIDA ROBUSTA: 
    Usa el prompt DETALLADO original para máxima precisión.
    Incluye limpieza inteligente para evitar errores si la IA habla de más.
    """
    if not sess_url:
        return {"Error": "Modelo de URL no ha sido cargado correctamente"}

    print(f"Analizando URL {url_usuario} con Cohere")
    
    # Prompt que le damos a Cohere para que obtenga los valores de la URL 
    prompt = f"""
    Actúa como un analista de ciberseguridad experto. Tu tarea es extraer características técnicas de la URL: "{url_usuario}" para alimentar un modelo de detección de phishing (Dataset UCI).

    Analiza la URL y genera un JSON con las siguientes 30 claves. 
    Usa los valores: -1 (Malicioso/Phishing), 0 (Sospechoso), 1 (Legítimo/Seguro).

    CLAVES Y GUÍA DE VALORES:
    1. "having_IP_Address": -1 si la URL usa una dirección IP en lugar de dominio, si no 1.
    2. "URL_Length": -1 si >75 caracteres, 0 si entre 54-75, 1 si <54.
    3. "Shortining_Service": -1 si usa acortadores (bit.ly, tinyurl), si no 1.
    4. "having_At_Symbol": -1 si tiene "@", si no 1.
    5. "double_slash_redirecting": -1 si hay "//" después de la posición 7 (http), si no 1.
    6. "Prefix_Suffix": -1 si el dominio tiene guiones (-), si no 1.
    7. "having_Sub_Domain": -1 si tiene más de 3 puntos (subdominios multiples), 0 si tiene 2, 1 si tiene 1 o ninguno.
    8. "SSLfinal_State": 1 si es HTTPS y certificado válido, -1 si es HTTP o certificado malo.
    9. "Domain_registeration_length": -1 si el dominio expira en < 1 año, 1 si > 1 año (Estímalo 1 si parece dominio corporativo serio).
    10. "Favicon": 1 si carga desde el mismo dominio, -1 si es externo.
    11. "port": 1 si usa puertos estándar (80, 443), -1 si usa puertos raros abiertos.
    12. "HTTPS_token": -1 si "https" aparece como parte del dominio (ej: https-google.com), si no 1.
    13. "Request_URL": -1 si >61% de objetos (imgs) son externos, 1 si <22%.
    14. "URL_of_Anchor": -1 si los enlaces <a> apuntan a otro dominio o vacíos, 1 si son internos.
    15. "Links_in_tags": -1 si tags Meta/Script/Link apuntan fuera, 1 si no.
    16. "SFH": -1 si el Server Form Handler está vacío o "about:blank", 1 si es válido.
    17. "Submitting_to_email": -1 si usa "mailto:", 1 si no.
    18. "Abnormal_URL": -1 si el host no está en la URL, 1 si normal.
    19. "Redirect": -1 si redirige > 4 veces, 1 si < 2.
    20. "on_mouseover": -1 si usa JS para cambiar status bar, 1 si no.
    21. "RightClick": -1 si deshabilita clic derecho, 1 si no.
    22. "popUpWidnow": -1 si abre popups con inputs, 1 si no.
    23. "Iframe": -1 si usa iframes invisibles, 1 si no.
    24. "age_of_domain": -1 si < 6 meses, 1 si > 6 meses.
    25. "DNSRecord": -1 si no tiene registro DNS (URL vacía), 1 si tiene.
    26. "web_traffic": 1 si es sitio popular (Alexa/SimilarWeb), -1 si no tiene tráfico o es nuevo.
    27. "Page_Rank": 1 si tiene PageRank alto, -1 si bajo.
    28. "Google_Index": 1 si está indexada en Google, -1 si no.
    29. "Links_pointing_to_page": 1 si tiene muchos backlinks, -1 si 0.
    30. "Statistical_report": -1 si la IP está en listas negras, 1 si limpia.

    Responde SOLO con el JSON válido. Sin explicaciones extra.
    """

    try:
        response = co.chat(
            model="command-r-08-2024", 
            messages=[{"role": "user", "content": prompt}]
        )
        
        texto_respuesta = response.message.content[0].text
        
        #Esto sirve para solucionar errores de URL inválida. Buscamos las llaves para saber donde empieza 
        #y donde acaba el Json que nos da Cohere
        inicio = texto_respuesta.find('{')
        fin = texto_respuesta.rfind('}') + 1

        if inicio != -1 and fin != -1:
            json_str = texto_respuesta[inicio:fin]
            features_json = json.loads(json_str)
        else:
            raise ValueError("La IA no devolvió un JSON limpio.")

        # Preparar vector para ONNX
        lista_valores = []
        for feature in FEATURES_ORDER:
            # Si falta alguna clave, asumimos 0 (sospechoso) por seguridad
            val = features_json.get(feature, 0)
            lista_valores.append(float(val))
            
        vector_np = np.array([lista_valores], dtype=np.float32)
        
        # Predicción ONNX
        prediccion = sess_url.run([label_name_url], {input_name_url: vector_np})[0]
        es_phishing = (prediccion[0] == 1)
        
        return {
            "es_phishing": es_phishing,
            "probabilidad": 95 if es_phishing else 5, 
            "tipo": "URL Web",
            "explicacion": "Análisis técnico profundo de 30 variables completado.",
            "detalles": features_json
        }

    except Exception as e:
        print(f"La IA tuvo problemas con esta URL compleja: {e}.")
        
        #Como mitigación de errores, en caso de que la URL no sea válida, damos esta salida
        #Esto es subjetivo, en nuestro caso hemos decidido que si la URL no es válida, en lugar de dar error
        #retornaremos que hay un 50% de que sea fraude, puesto que no lo sabemos.
        return {
            "es_phishing": False, #Da igual
            "probabilidad": 50, 
            "tipo": "URL Web",
            "explicacion": "La URL es demasiado compleja para el análisis automático estándar. Se recomienda revisar manualmente.",
            "detalles": {"nota": "Análisis parcial debido a complejidad de la URL"}
        }

def guardar_prediccion(tipo, input_usuario, resultado):
    collection = db["predictions_history"]

    documento = {
        "tipo": tipo,
        "input": input_usuario,
        "es_phishing": bool(resultado.get("es_phishing")),
        "probabilidad": float(resultado.get("probabilidad")) if resultado.get("probabilidad") is not None else None,
        "explicacion": resultado.get("explicacion"),
        "fecha": datetime.now(timezone.utc)
    }

    collection.insert_one(documento)


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

def fig_to_base64(fig):
    """
    Convierte una figura de Matplotlib en PNG (base64)
    para poder incrustarla directamente en HTML.
    """
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")        

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

    predictions = list(
        db["predictions_history"].find().sort("fecha", -1)
    )

    antivirus = list(
        db["best antivirus"].find({}, {"_id": 0})
    )

    mapa = list(
        db["mapa_phishing"].find({}, {"_id": 0})
    )

    minigame = list(
        db["minigame"].find({}, {"_id": 0})
    )

    collection = db["predictions_history"]
    datos = list(collection.find().sort("fecha", -1))

    return render_template(
        "history.html",
        predictions_history=predictions,
        best_antivirus=antivirus,
        mapo_phishing=mapa,
        minigame=minigame
    )

@app.route("/stats", methods=["GET"])
def stats():
    return render_template("stats.html")

@app.route("/stats/data", methods=["GET"])
def stats_data():
    # ---------- 1) Dataset del minijuego ----------
    images = list(db["minigame"].find({}, {"_id": 0, "is_phishing": 1}))
    n_images = len(images)
    n_phish = sum(1 for x in images if x.get("is_phishing") is True)
    n_legit = n_images - n_phish

    dataset = {
        "n_images": n_images,
        "n_phish": n_phish,
        "n_legit": n_legit
    }

    # ---------- 2) Intentos del minijuego (accuracy por día) ----------
    attempts = list(db["minigame_attempts"].find({}, {"_id": 0, "ts": 1, "success": 1}))
    minigame_summary = None
    daily_accuracy = {"labels": [], "values": []}

    if attempts:
        dfA = pd.DataFrame(attempts)
        dfA["ts"] = pd.to_datetime(dfA["ts"], errors="coerce")
        dfA = dfA.dropna(subset=["ts"])
        dfA["date"] = dfA["ts"].dt.date
        dfA["success"] = dfA["success"].astype(bool)

        minigame_summary = {
            "total_attempts": int(len(dfA)),
            "accuracy_pct": round(float(dfA["success"].mean() * 100), 1)
        }

        by_day = dfA.groupby("date")["success"].mean().reset_index()
        daily_accuracy["labels"] = [str(d) for d in by_day["date"].tolist()]
        daily_accuracy["values"] = [round(float(v * 100), 1) for v in by_day["success"].tolist()]

    # ---------- 3) Predicciones Cohere ----------
    preds = list(db["cohere_predictions"].find({}, {"_id": 0, "result": 1}))
    cohere_summary = None

    cohere_msg_counts = {
        "labels": ["No es mensaje", "Es mensaje"],
        "values": [0, 0]
    }

    cohere_prob_hist = {
        "labels": [f"{i*10}-{i*10+9}" for i in range(10)],
        "counts": [0] * 10
    }

    if preds:
        rows = []
        for p in preds:
            r = p.get("result") or {}
            rows.append({
                "es_mensaje": r.get("es_mensaje"),
                "es_phishing": r.get("es_phishing"),
                "probabilidad_phishing": r.get("probabilidad_phishing")
            })

        dfP = pd.DataFrame(rows)
        dfP["es_mensaje"] = dfP["es_mensaje"].astype("boolean")
        dfP["probabilidad_phishing"] = pd.to_numeric(
            dfP["probabilidad_phishing"], errors="coerce"
        )

        total_preds = int(len(dfP))
        mensajes = int(dfP["es_mensaje"].fillna(False).sum())
        no_mensajes = total_preds - mensajes

        cohere_msg_counts["values"] = [no_mensajes, mensajes]

        df_msg = dfP[dfP["es_mensaje"] == True].copy()
        phishing = int((df_msg["es_phishing"] == True).sum())
        legit = int((df_msg["es_phishing"] == False).sum())

        cohere_summary = {
            "total_preds": total_preds,
            "mensajes": mensajes,
            "no_mensajes": no_mensajes,
            "phishing_en_mensajes": phishing,
            "legit_en_mensajes": legit
        }

        # Histograma de probabilidad (bins de 10%)
        probs = pd.to_numeric(
            df_msg["probabilidad_phishing"], errors="coerce"
        ).dropna()
        probs = probs.clip(lower=0, upper=100)

        for v in probs.tolist():
            idx = min(int(v // 10), 9)
            cohere_prob_hist["counts"][idx] += 1

    return {
        "dataset": dataset,
        "minigame_summary": minigame_summary,
        "daily_accuracy": daily_accuracy,
        "cohere_summary": cohere_summary,
        "cohere_msg_counts": cohere_msg_counts,
        "cohere_prob_hist": cohere_prob_hist
    }


@app.route('/report', methods=['GET', 'POST'])
def report():
    """
    Formulario que tiene que rellenar el usuario para poder reportar el intento de phishing.
    Cada vez que el usuario envía algo tiene que subirse a la db
    """
    collection_links = db["report_reliable_links"]
    collection_antivirus = db["best antivirus"]

    report_links = collection_links.find_one({}, {"_id": 0})
    antivirus_list = list(collection_antivirus.find({}, {"_id": 0}))

    resultados = None
    tipo_seleccionado = None
    pais_seleccionado = None

    if request.method == "POST":
        tipo_seleccionado = request.form.get("tipo")
        pais_seleccionado = request.form.get("pais", "Global")

        if tipo_seleccionado:
            resultados = report_links.get(tipo_seleccionado, {}).get(
                pais_seleccionado,
                report_links.get(tipo_seleccionado, {}).get("Global", [])
            )

    return render_template(
        "report.html",
        resultados=resultados,
        antivirus_list=antivirus_list,
        tipo_seleccionado=tipo_seleccionado,
        pais_seleccionado=pais_seleccionado
    )



@app.route("/api/mapa-phishing")
def mapa_phishing():
    """
    Devuelve los datos de phishing para el mapa interactivo
    """
    collection = db["mapa_phishing"]

    datos = list(collection.find({}, {
        "_id": 0,          #quitamos el ObjectId
        "pais": 1,
        "lat": 1,
        "lon": 1,
        "tipo": 1,
        "titulo": 1,
        "historia": 1,
        "anio": 1
    }))

    return jsonify(datos)

@app.route("/mapa")
def mapa():
    """
    Página del mapa interactivo de phishing
    """
    return render_template("mapa_phishing.html")
   

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    """
    Da las opciones de predecir por URL de la imagen, texto o URL de página web dudosa

    Maneja 3 casos:
    1. img_url / img_file (función analisis_img) 
    2. text_input (función analisis_texto_onnx) 
    3. url_input (función analisis_url_hibrido) 

    A partir de una URL a la imagen, Cohere detecta si es un mensaje. En caso de que lo sea
    identifica si se trata de phishing o no y su porcentaje de phishing, por lo contrario envía un mensaje
    al usuario explicando que la imagen no es válida para la predicción
    """
    resultado_url = None
    resultado_img = None
    resultado_texto = None

    if request.method == "POST":
        # Caso 1.1: URL de Imagen
        img_url = request.form.get("img_url")
        if img_url:
            res = analisis_img(img_url)
            if res:
                resultado_img = res

                guardar_prediccion(
                    tipo="imagen",
                    input_usuario=img_url,
                    resultado={
                        "es_phishing": res.get("es_phishing"),
                        "probabilidad": res.get("probabilidad_phishing"),
                        "explicacion": res.get("explicacion")
                    }
                )
            if res is None:
                resultado_img = {"error": "Error al procesar la imagen, pruebe con otra."}
            else:
                resultado_img = res

        # Caso 1.2: (Subir el archivo de la imagen)
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

        # Caso 2: URL de una web
        url_input = request.form.get("url_input")
        if url_input:
            resultado_url = analisis_url_hibrido(url_input)
            if resultado_url and "error" not in resultado_url:
                guardar_prediccion(
                    tipo="url",
                    input_usuario=url_input,
                    resultado=resultado_url
                )


        # Caso 3: El texto de un mensaje
        text_input = request.form.get("text_input")
        if text_input:
            resultado_texto = analisis_texto_onnx(text_input)
            if resultado_texto and "error" not in resultado_texto:
                guardar_prediccion(
                    tipo="texto",
                    input_usuario=text_input,
                    resultado=resultado_texto
                )


            # ✅ AQUÍ EXACTAMENTE: guardar predicción de archivo
            try:
                db["cohere_predictions"].insert_one({
                    "ts": datetime.utcnow(),
                    "source_type": "file",
                    "source": img_file.filename,
                    "result": res
                })
            except Exception as e:
                print("Error guardando predicción Cohere (file):", e)

    return render_template(
        "predictions.html",
        resultado_url=resultado_url,
        resultado_img=resultado_img,
        resultado_texto=resultado_texto
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

        from datetime import datetime
        try:
            db["minigame_attempts"].insert_one({
                "ts": datetime.utcnow(),
                "image_id": ObjectId(image_id),
                "user_answer": bool(user_answer_bool),
                "correct_answer": bool(image["is_phishing"]),
                "success": bool(result["correct"])
            })
        except Exception as e:
            print("Error guardando intento del minijuego:", e)

    else:
        image = random.choice(images)
        image_id = str(image["_id"])

    return render_template(
        "minigame.html",
        image_id=image_id,
        result=result
    )

@app.route('/presentacion', methods=['GET', 'POST'])
def presentacion():
    return render_template("presentacion.html")

if __name__ == "__main__":
    app.run(debug = True, host = "localhost", port  = 5000)