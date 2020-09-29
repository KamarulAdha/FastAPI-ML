from fastapi import FastAPI
from ml import nlp
from pydantic import BaseModel
from typing import List

"""app is the application we are creating, which is an  instance of FastAPI class"""
app = FastAPI()


"""
1) @app ==> decorator
2) "/" will redirect to the root path and perform a HTTP GET request
"""
@app.get("/")
def read_main():
    return {'message': 'Hello World'}


class Article(BaseModel):
    content: str

@app.post("/article/")
def analyze_article(articles: List[Article]):
    """
    Analyze an article and extract entities with ⚡spaCy⚡

    Statistical models *will* have **errors**.

    * Extract entities
    * Display comments
    """
    ents = []
    for article in articles:
        doc = nlp(article.content)
        for ent in doc.ents:
            ents.append({"entity": ent.text, "label": ent.label_})

    return {"entities": ents}
