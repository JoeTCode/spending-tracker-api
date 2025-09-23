from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from utils.barclays.format_csv import format_memo
import os
from config import RULES_PATH, CASE_SENSITIVE_RULES_PATH, CATEGORIES
import pandas as pd
import numpy as np
import tensorflow as tf

script_dir = os.path.dirname(__file__)  # path of current script
SAVE_NAME = "model_"
SAVE_FOLDER_PATH = os.path.join(script_dir, "models/mlp")
EXTENSION = ".h5"

def save_model(model, folder_path, EXTENSION=EXTENSION, SAVE_NAME=SAVE_NAME):
    files = os.listdir(folder_path)
    files = [os.path.splitext(file)[0] for file in files if not file.startswith('.')]
    filename = SAVE_NAME
    
    num = 0
    if len(files) > 0:
        for file in files:
            num = max(num, int(file.split('_')[-1]) + 1)

    filename += str(num)
    if EXTENSION: filename += EXTENSION

    folderpath = os.path.join(folder_path, filename)
    # Convert to forward slashes
    folderpath = folderpath.replace("\\", "/")

    model.save(folderpath)

    return folderpath


def get_latest_model(folder_path, EXTENSION=EXTENSION, SAVE_NAME=SAVE_NAME):
    files = os.listdir(folder_path)
    files = [os.path.splitext(file)[0] for file in files if not file.startswith('.')]
    filename = SAVE_NAME

    num = 0
    if len(files) > 0:
        for file in files:
            num = max(num, int(file.split('_')[-1]))

    filename += str(num)
    if EXTENSION: filename += EXTENSION
    folderpath = os.path.join(folder_path, filename)
    
    # Convert to forward slashes
    folderpath = folderpath.replace("\\", "/")

    return folderpath


idx_to_labels = { k : v for k, v in enumerate(CATEGORIES)}
labels_to_idx = { k : v for v, k in enumerate(CATEGORIES)}
BERT_DIM = 384
NUM_LABELS = len(CATEGORIES)

# Load model on start
MODEL_PATH = get_latest_model(SAVE_FOLDER_PATH)
print(f"Model loaded from: {MODEL_PATH}")
MLP_MODEL = tf.keras.models.load_model(MODEL_PATH)


def predict(model, embeddings):
    predictions = model.predict(embeddings)
    maxProbs = predictions.max(-1)

    predicted_indices = np.argmax(predictions, axis=-1)
    predicted_categories = [idx_to_labels[idx] for idx in predicted_indices]

    return predicted_categories, maxProbs


# Load rules
rules = pd.read_csv(RULES_PATH)
case_sensitive_rules = pd.read_csv(CASE_SENSITIVE_RULES_PATH)

class PredictItems(BaseModel):
    embeddings: List[List[float]] # list of embedding vectors
    descriptions: List[str]

# Predict request schema
class PredictRequest(BaseModel):
    predict_data: PredictItems

# Response schema
class Prediction(BaseModel):
    predicted_category: str
    confidence: float

# Training request schema
class TrainItems(BaseModel):
    embeddings: List[List[float]] # list of embedding vectors
    categories: List[str]

class TrainRequest(BaseModel):
    train_data: TrainItems

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=List[Prediction])
def predict_path(request: PredictRequest):
    # Extract memos
    # descriptions = [format_memo(t.description, False) for t in request.transactions]
    embeddings = np.array(request.predict_data.embeddings, dtype="float32")
    descriptions = request.predict_data.descriptions

    # Predict
    preds, probabilities = predict(MLP_MODEL, embeddings)

    predicted_categories = []
    for i, d in enumerate(descriptions):
        
        matches = case_sensitive_rules[
            case_sensitive_rules["company_name"].apply(lambda x: x in d)
        ]
        if not matches.empty:
            predicted_categories.append(matches.iloc[0]["category"])
            probabilities[i] = 1.0
            continue

        matches = rules[
            rules["company_name"].apply(lambda x: x in d)
        ]
        if not matches.empty:
            predicted_categories.append(matches.iloc[0]["category"])
            probabilities[i] = 1.0
            continue

        predicted_categories.append(preds[i])
        
    # Build response
    results = [
        Prediction(predicted_category=p, confidence=c)
        for p, c in zip(predicted_categories, probabilities)
    ]

    return results


@app.post("/train")
def train_route(request: TrainRequest):
    # Use globally declared model
    global MLP_MODEL

    MLP_MODEL.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    embeddings = tf.convert_to_tensor(request.train_data.embeddings, dtype=tf.float32)
    categories = request.train_data.categories

    labels = np.array([labels_to_idx[label] for label in categories])

    MLP_MODEL.fit(
        embeddings, 
        labels,
        epochs=1,
        batch_size=16,
    )

    path = save_model(MLP_MODEL, SAVE_FOLDER_PATH)
    print(f"Model saved at: {path}")

    return {"status": "model updated"}