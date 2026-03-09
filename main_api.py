from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from transformer_model import TransformerClassifier

app = FastAPI(title="Fraud Detection API", version="1.0")

# Define input schema
class SequenceData(BaseModel):
    sequence: list[list[float]]  # shape: (20, 5)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier()
model.load_state_dict(torch.load("transformer_fraud_model.pth", map_location=device))
model = model.to(device)
model.eval()

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(data: SequenceData):
    # Convert input to tensor
    X = torch.tensor([data.sequence], dtype=torch.float32).to(device)

    # Run model
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)

    label_map = {0: "normal user", 1: "bot", 2: "fraudster"}
    return {"prediction": int(predicted.item()), "class_name": label_map[int(predicted.item())]}
