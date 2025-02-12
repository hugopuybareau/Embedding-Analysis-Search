# Imports

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from embedding_models import EmbeddingModel

app = FastAPI() # Initialize the API

model = EmbeddingModel(method='sbert') # Initialize the sbert model

texts = [ # The text corpus
    "Artificial intelligence is transforming finance.",
    "Stock markets drop after the FED announcement.",
    "Tesla unveils a new battery with record range.",
    "Apple launches iPhone 16 with new features.",
    "The job market in France is improving.",
    "AI and cybersecurity are reshaping data protection."
    "Artificial intelligence is transforming economy.",
]

model.fit(texts) # Train and Index the texts
model.index_texts(texts)

class QueryRequest(BaseModel): # Pydantic model for the query
    query: str
    how_much_results: int = 3

class DocumentRequest(BaseModel): 
    texts: List[str]

@app.get("/") #GET route
def home():
    return {"message": "API running"}

@app.post("/search/") # Post endpoint because user sends input data
def search(request: QueryRequest):
    results = model.search_similar(request.query, request.how_much_results)
    return {"query": request.query, "results": [{"document": doc, "score": f"{float(score):.2f}, "} for doc, score in results]}

@app.post("/index/") # Post endpoint because user sends data again
def index_new_texts(request: DocumentRequest):
    model.index_texts(request.texts)
    return {"message": f"Indexed {len(request.texts)} new texts."}

