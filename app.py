from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import re
import numpy as np

app = FastAPI(title="Personalized Medicine API")

# Load model (upload your files here)
model = torch.load('model.pth', map_location='cpu')
model.eval()
processor = joblib.load('text_processor.pkl')
le_gene = joblib.load('le_gene.pkl')
le_var = joblib.load('le_variation.pkl')

class Input(BaseModel):
    gene: str
    variation: str
    text: str

def preprocess(text):
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    indices = [processor['word2idx'].get(w, 1) for w in text.split()[:256]]
    return indices + [0] * (256 - len(indices))

@app.post("/predict")
def predict(data: Input):
    text_idx = preprocess(data.text)
    try:
        gene_enc = le_gene.transform([data.gene])[0]
        var_enc = le_var.transform([data.variation])[0]
    except:
        return {"error": "Invalid gene or variation"}
    
    tab = torch.tensor([[float(gene_enc), float(var_enc)]])
    text_t = torch.tensor([text_idx], dtype=torch.long)
    
    with torch.no_grad():
        out = model(text_t, tab)
        prob = torch.softmax(out, 1)[0]
        pred = int(prob.argmax().item()) + 1
        conf = float(prob.max().item())
    
    return {
        "prediction": pred,
        "confidence": conf,
        "interpretation": f"Class {pred}: {'High-impact driver mutation – Recommend targeted therapy!' if pred <=3 else 'Low-impact passenger mutation – Standard treatment.'}"
    }

@app.get("/")
def root():
    return {"message": "API Ready! POST to /predict"}
