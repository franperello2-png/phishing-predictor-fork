import os
import sys
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from onnxmltools import convert_sparkml
from onnxmltools.convert.common.data_types import FloatTensorType
from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
import onnx

# --- 1. CONFIGURACIÓN CRÍTICA PARA WINDOWS ---
# Obtenemos la ruta real del python de tu entorno bigdata_env
python_path = sys.executable
os.environ['PYSPARK_PYTHON'] = python_path
os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

# Configuramos la sesión para que NO use los sockets de Unix y use el sistema de archivos local "crudo"
spark = (
    SparkSession.builder
    .appName("Fraude_To_ONNX")
    .master("local[*]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    # Este parámetro es clave para saltarse winutils al crear carpetas temporales
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    .getOrCreate()
)

# Limpiamos los logs para ver solo lo importante
spark.sparkContext.setLogLevel("ERROR")

print(f">>> Usando Python desde: {python_path}")

try:
    # 2. Dataset de Fraude (Simulando el formato del Titanic/Pingüinos)
    # En un caso real, aquí cargarías: spark.read.csv("tus_datos.csv")
    data = {
        "longitud_texto": [10.0, 200.0, 15.0, 150.0, 12.0, 300.0],
        "tiene_enlace": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "palabras_alerta": [0.0, 5.0, 0.0, 3.0, 1.0, 8.0],
        "es_fraude": [0, 1, 0, 1, 0, 1] 
    }
    pdf = pd.DataFrame(data)
    df = spark.createDataFrame(pdf)

    # 3. Preprocesado
    df_proc = df.withColumn("label", col("es_fraude").cast("int"))

    # 4. Features (3 variables de entrada)
    feature_cols = ["longitud_texto", "tiene_enlace", "palabras_alerta"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_features = assembler.transform(df_proc)

    # 5. Entrenamiento del modelo
    print("Entrenando RandomForest...")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10)
    rf_model = rf.fit(train_features)
    print("✅ Modelo entrenado.")

    # 6. Conversión a ONNX
    # IMPORTANTE: Aquí onnxmltools creará una carpeta temporal
    initial_types = [("features", FloatTensorType([None, 3]))]
    
    onnx_model = convert_sparkml(
        rf_model,
        name="DetectorFraudeONNX",
        initial_types=initial_types,
        spark_session=spark,
        target_opset=min(DEFAULT_OPSET_NUMBER, onnx_opset_version())
    )

    # 7. Guardar el archivo final
    onnx.save(onnx_model, "modelo_fraude.onnx")
    print(f"🚀 ¡ÉXITO! Archivo 'modelo_fraude.onnx' generado en la carpeta actual.")

except Exception as e:
    print(f"❌ Error durante el proceso: {e}")
    print("\nNOTA: Si el error menciona 'winutils.exe', significa que onnxmltools")
    print("necesita ese archivo de 2MB para escribir el ONNX en Windows.")

finally:
    spark.stop()