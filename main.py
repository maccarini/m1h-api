import jwt
from fastapi import FastAPI, Header, Request, Depends, Response, Form
from typing import Optional
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File, Body
import io

from bertopic.dimensionality import BaseDimensionalityReduction
import nltk
from nltk.corpus import stopwords
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from nltk.util import skipgrams
import functools
from nltk import word_tokenize
from transformers.pipelines import pipeline
from config import settings
import openai
from bertopic.backend import OpenAIBackend
import tiktoken

OPENAI_KEY = settings.openai_api_key
openai.api_key = OPENAI_KEY

JWT_SECRET = settings.m1d_api_token
JWT_ALGORITHM = "HS256"

app = FastAPI()


origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://m1d.wholemeaning.com"
]

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
STOPWORDS = list(stopwords.words("spanish"))
relevant = {"no", "ni", "sin", "nada", "sí", "estad", "fuera"}
[STOPWORDS.remove(word) for word in relevant]
UMAP_N_COMPONENTS = 5
DBSCAN_MIN_CLUSTER_SIZE =  30
N_NEIGHBORS = 5 
embedding_model = OpenAIBackend(embedding_model="text-embedding-ada-002", batch_size=512)
umap_model = UMAP(n_neighbors=N_NEIGHBORS, n_components=UMAP_N_COMPONENTS, min_dist=0.0, metric='cosine')
hdbscan_model = HDBSCAN(min_cluster_size=DBSCAN_MIN_CLUSTER_SIZE, min_samples=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model = CountVectorizer(ngram_range = (2,3), stop_words=STOPWORDS)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

async def get_body(request: Request):
    return await request.body()

@app.get("/healthz", status_code=200)
def healthz():
    return {"status": "ok"}


@app.post("/clusterize")
async def clusterize(
    csv_file: Optional[UploadFile] = Form(None),
    authorization: str = Header(None)
):
    try:
        decoded = secure(authorization)
    except:
        return "Acceso Denegado."
    try:
        df = pd.read_csv(csv_file.file)
        docs = df["tematica"].tolist()
        topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=True,
        #nr_topics = "auto",
        )
        topics, probs = topic_model.fit_transform(docs)
        freq = topic_model.get_topic_info()
        print(f'Se ha generado un output de {len(freq)} tópicos.')
        return {
            "tags": topic_model.topics_,
            "topic_info":freq.to_json()
        }
    except Exception as e:
        print(e)
        return "Se ha producido un error en la ejecución."


def secure(token):
    decoded_token = jwt.decode(token, JWT_SECRET, algorithms=JWT_ALGORITHM)

    return decoded_token

