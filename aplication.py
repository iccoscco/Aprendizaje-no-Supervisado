import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Descargar el conjunto de stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Función para preprocesar y limpiar el texto
def preprocess_text(text):
    # Eliminar caracteres no alfabéticos
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convertir a minúsculas
    text = text.lower()
    # Tokenización
    tokens = word_tokenize(text)
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Aplicar stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    # Reconstruir el texto preprocesado
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Datos de ejemplo (mensajes de redes sociales etiquetados con sentimientos)
data = [
    {"text": "I love this product!", "sentiment": "positive"},
    {"text": "Terrible experience with customer service.", "sentiment": "negative"},
    # ... más mensajes ...
]

# Preprocesar y limpiar los mensajes
preprocessed_data = [preprocess_text(entry["text"]) for entry in data]

# Crear la representación vectorial de los datos utilizando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_data)

# Calcular la similitud de coseno entre los vectores
cosine_similarities = cosine_similarity(X, X)

# Aplicar DBSCAN para agrupar mensajes similares
dbscan = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
clusters = dbscan.fit_predict(cosine_similarities)

# Agregar la información de agrupación a los datos originales
for i, entry in enumerate(data):
    entry["cluster"] = clusters[i]

# Imprimir los resultados
for entry in data:
    print(f"Text: {entry['text']}, Sentiment: {entry['sentiment']}, Cluster: {entry['cluster']}")
