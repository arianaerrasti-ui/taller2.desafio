import pandas as pd

#cargamos el datset preprocesado
df=pd.read_csv("amazon_reviews_preprocesado.csv")
#Eliminamos las reseñas vacías
df = df.dropna(subset=["text_preprocesado"]).copy()



#-------------SEGUNDA ENTREGA---------------------------------

# Bag of Words

# Cargamos el dataset preprocesado (reseñas limpias en formato cadena de texto)
df = pd.read_csv("amazon_reviews_preprocesado.csv")
#importamos la libreria
from sklearn.feature_extraction.text import CountVectorizer

# Crear el vectorizador (toma como parametro una cadena de texto y convierte texto a numeros)
#vectorizador_bow variable que almacena el objeto countvectorizar
vectorizador_bow = CountVectorizer()

# Generar la matriz numérica (X_bow) a partir del texto preprocesado (text_preprocesado)
#fit aprende el vocabulario total de las reseñas y transform utiliza ese vocab y transforma cada reseña en una representacion numerica
#fillna remplaza las palabras faltantes con cadenas vacias
X_bow = vectorizador_bow.fit_transform(
    df["text_preprocesado"].fillna("")
)


#----------------------------------------------------------------------
#TF-IDF con unigramas (terminos individuales)
#le asigna un peso segun su frecuencia a cada reseña y q tan comun es en el conjunto total de rese

#importamos nueva herramienta
from sklearn.feature_extraction.text import TfidfVectorizer

# Creamos el vectorizador y lo almacenamos en la variable vectorizador_tfidf_uni
vectorizador_tfidf_uni = TfidfVectorizer()

# Generar la matriz numérica (al igual q lo hicimos en bag of words)
#ahora fit_transform aprende el vocabulario total (fit) y calcula pesos de las palabras en cada reseña(tranform)
X_tfidf_uni = vectorizador_tfidf_uni.fit_transform(
    df["text_preprocesado"].fillna("")
)

#TF-IDF-bigrama(pares de palabras(not good,very nice, etc))

# Creamos el vectorizador para bigramas
vectorizador_tfidf_bi = TfidfVectorizer(ngram_range=(1, 2))

# Generamos la matriz numérica
X_tfidf_big = vectorizador_tfidf_bi.fit_transform(
    df["text_preprocesado"].fillna("")
)


#MOSTRAMOS RESULTADOS:

# Cantidad de reseñas y términos (son las mismas que en tf-idf)
print("Cantidad total de reseñas:", X_bow.shape[0])#filas
print("Tamaño del vocabulario BoW:", X_bow.shape[1])#columnas
print("Dimensiones de X_bow:", X_bow.shape) #(filas,columnas)

# Mostrar las palabras y sus conteos en la primera reseña (para ver q funciona)
print("\nPalabras y conteos de la primera reseña:")

conteos = X_bow[0].toarray().flatten()
palabras = vectorizador_bow.get_feature_names_out()

for i in conteos.nonzero()[0]:
    print(palabras[i], ":", conteos[i])
    
    
# Pesos de las palabras de la primera reseña (unigramas):
pesos = X_tfidf_uni[0].toarray().flatten()
vocabulario_tfidf = vectorizador_tfidf_uni.get_feature_names_out()

print("\nPalabras y pesos TF-IDF de la primera reseña:")

for palabra, peso in zip(vocabulario_tfidf, pesos):

    if peso > 0:

        print(palabra, ":", round(peso, 4))
        
#Pesos primera reseña (comparacion unigramos vs bigramas):

pesos = X_tfidf_big[0].toarray().flatten()
vocabulario_tfidf = vectorizador_tfidf_bi.get_feature_names_out()

print("\nPalabras y pesos TF-IDF de la primera reseña (bigramas):")

for palabra, peso in zip(vocabulario_tfidf, pesos):
    if peso > 0:
        print(palabra, ":", round(peso, 4))
        
    #PENDIENTE---------------------------------------------------------    
#EMBEDDINGS=Buscamos representar relaciones sistematicas, convierte texto a numeros con un modelo preentrenado
#BoW=cuantas veces aparece una palabra en una reseña del vocab total 
#TF-ID=calcula el peso de cada palabra del vocabulario segun la frecuencia en la reseñas

#importamos biblioteca (no pudimos usar gensim)
from sentence_transformers import SentenceTransformer

 
 #sentencetransformer=carga un modelo (all-Mini..) preentrenado q transforma texto en vectores
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

#Generamos embeddings de las reseñas(estan en text procesado del dataframe df)
textos = df["text_preprocesado"].fillna("").tolist() #defino textos
 #resultado del procesamiento=embeddings
 
embeddings = modelo_embeddings.encode( #encode instruccion principal
    textos, #lista de reseñas
    batch_size=8, #procesa reseñas de a lotes de a 8
    show_progress_bar=True, #muestra el progreso del procesamiento del modelo
convert_to_numpy=True #el resultado es una matriz (filas\reseñas,columnas\valores numericos obtenidos)
)

print("Dimensiones de los embeddings:", embeddings.shape)
#----------------------------------------------------------------------------------

