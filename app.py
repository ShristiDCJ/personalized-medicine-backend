from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import re
import numpy as np

app = FastAPI(title="Personalized Medicine API")

# Load model and processors
model = torch.load('model.pth', map_location='cpu')
model.eval()
processor = joblib.load('text_processor.pkl')
le_gene = joblib.load('le_gene.pkl')
le_var = joblib.load('le_variation.pkl')

class InputData(BaseModel):
    gene: str
    variation: str
    text: str

def preprocess(text: str) -> list:
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    words = text.split()[:256]
    indices = [processor['word2idx'].get(w, 1) for w in words]
    return indices + [0] * (256 - len(indices))

@app.post("/predict")
def predict(data: InputData):
    try:
        text_idx = preprocess(data.text)
        gene_enc = le_gene.transform([data.gene])[0]
        var_enc = le_var.transform([data.variation])[0]
    except Exception as e:
        return {"error": "Invalid gene or variation", "details": str(e)}
    
    tab = torch.tensor([[float(gene_enc), float(var_enc)]], dtype=torch.float)
    text_tensor = torch.tensor([text_idx], dtype=torch.long)
    
    with torch.no_grad():
        out = model(text_tensor, tab)
        prob = torch.softmax(out, dim=1)[0]
        pred = int(prob.argmax().item()) + 1
        conf = float(prob.max().item())
    
    return {
        "prediction": pred,
        "confidence": round(conf, 4),
        "interpretation": (
            f"Class {pred}: High-impact driver mutation – Recommend targeted therapy!"
            if pred <= 3 else
            f"Class {pred}: Low-impact passenger mutation – Standard treatment."
        )
    }

@app.get("/")
def root():
    return {"message": "Personalized Medicine API is running!"}
