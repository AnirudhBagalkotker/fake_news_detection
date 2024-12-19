from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from newspaper import Article
import os
import json

import numpy as np
from importModel import Hybrid, preprocess, padArticles
import torch.nn.functional as F

app = FastAPI()
frontend_build_dir = os.path.join(os.getcwd(), "../frontend", "build")
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(frontend_build_dir, "static")),
    name="static",
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embeddingDim = 300
hiddenDim = 256
VocabSize = 33844
GloveEmbeddings = np.load("glove_embeddings.npy")
model = Hybrid(embeddingDim, VocabSize, hiddenDim, embeddings=GloveEmbeddings).to(
    device
)
checkpoint = torch.load("Model_Hybrid_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

with open("vocab.json", "r") as f:
    vocab = json.load(f)


class ArticleRequest(BaseModel):
    url: str


class TextRequest(BaseModel):
    title: str
    text: str


@app.get("/")
async def home():
    return FileResponse(os.path.join(frontend_build_dir, "index.html"))


def extract_article(url: str):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract article: {e}")


def predict(title, text):
    text = title + ". " + text

    preprocessed_text = preprocess(text)
    padded_text = padArticles([preprocessed_text], vocab=vocab)

    if padded_text.shape[1] < 16:
        padding = 16 - padded_text.shape[1]
        padded_text = F.pad(padded_text, (0, padding))

    input_tensor = padded_text.to(device)

    with torch.no_grad():
        model.eval()
        output = model(input_tensor)

    prediction = torch.sigmoid(output).item()
    result = "True" if prediction > 0.5 else "Fake"

    print("prediction:", prediction)

    return result


# Prediction route
@app.post("/predict")
def predict_url(request: ArticleRequest):
    try:
        title, text = extract_article(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract article: {e}")

    result = predict(title, text)

    return {"title": title, "prediction": result}


@app.post("/prediction")
def predict_article(request: TextRequest):
    title = request.title
    text = request.text

    result = predict(title, text)
    return {"title": title, "prediction": result}


# To run the app, use the command below:
# uvicorn main:app --reload
