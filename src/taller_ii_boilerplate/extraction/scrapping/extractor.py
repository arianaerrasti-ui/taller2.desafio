
import os
import time
import requests
import pandas as pd

#EXTRAEMOS LOS DATOS DE LA API

URL_API = "https://amazon-reviews-api-g5ae.onrender.com/reviews"

#cantidad total de reseñas
TOTAL = 210000
LIMITE = 100
#archivo donde se cargara el dataset
ARCHIVO = "amazon_reviews_extraido.csv"


# Tuvimos problemas para cargarlo
if os.path.exists(ARCHIVO):
    df_existente = pd.read_csv(ARCHIVO)
    inicio = len(df_existente)
else:
    inicio = 0

print(f"Reseñas ya guardadas: {inicio}")
print(f"Faltan: {TOTAL - inicio}")

# Continuar desde donde quedo
# Recorre la API de 100 a 100 reseñas
for offset in range(inicio, TOTAL, LIMITE):

    

        respuesta = requests.get(
            URL_API,
            params={
                "offset": offset,
                "limit": LIMITE
            },
            timeout=30
        )
#verifico
        respuesta.raise_for_status()
#convertimos la respuesta en un diccionario
        datos = respuesta.json()
#lista de reseñas
        resenas = datos["data"]
#creamos el data frame
        df_lote = pd.DataFrame(resenas)

        # Agregar las nuevas reseñas (las faltantes) al CSV
        df_lote.to_csv(
            ARCHIVO,
            mode="a",
            header=False,
            index=False
        )

        print(f"Guardadas: {offset + len(resenas)} / {TOTAL}")

        time.sleep(1)

print("Proceso terminado.")




# Cargamos el dataset
df = pd.read_csv("amazon_reviews_extraido.csv")

# Mostramos informacion
print("Cantidad de filas:", len(df))
print("Cantidad de columnas:", len(df.columns))

print("\nColumnas:")
print(df.columns.tolist())

print("\nPrimeras 5 filas:")
print(df.head())

print("\nDatos faltantes:")
print(df.isnull().sum())

print("\nDuplicados:")
print(df.duplicated().sum())

print("\nDistribución de etiquetas:")
print(df["label"].value_counts())

# PREPROCESAMIENTO

#importamos librerias
import string
import nltk

from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# FUNCIÓN DE PREPROCESAMIENTO


stop_words = stopwords.words("english")

#definimos funcion
def preprocesar(texto):

    texto = texto.lower()

    # Eliminar números
    texto_sin_numeros = ""

    for caracter in texto:
        if not caracter.isdigit():
            texto_sin_numeros = texto_sin_numeros + caracter

    texto = texto_sin_numeros

    # Eliminar signos de puntuación
    texto = texto.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenizar
    tokens = word_tokenize(texto)

    # Eliminar stopwords
    tokens_limpios = []

    for token in tokens:
        if token not in stop_words:
            tokens_limpios.append(token)

    # Eliminar tokens de un solo carácter
    tokens_finales = []

    for token in tokens_limpios:
        if len(token) > 1:
            tokens_finales.append(token)

    return tokens_finales


# Aplicar el preprocesamiento al dataset

df["text_preprocesado"] = df["text"].apply(preprocesar)


# Mostrar las primeras 5 reseñas
# para comprobar el resultado

print(df[["text", "text_preprocesado"]].head())
# Guardamos el dataset preprocesado
df.to_csv("amazon_reviews_preprocesado.csv", index=False)

print("Dataset preprocesado guardado.")