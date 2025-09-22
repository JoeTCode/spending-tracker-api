from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
# import joblib
from sentence_transformers import SentenceTransformer
from utils.barclays.format_csv import format_memo
import os
from config import RULES_PATH, CASE_SENSITIVE_RULES_PATH, CATEGORIES
import pandas as pd
import numpy as np
import tensorflow as tf

script_dir = os.path.dirname(__file__)  # path of current script
SAVE_NAME = "model_"
SAVE_FOLDER_PATH = os.path.join(script_dir, "models/mlp")


def save_model(model, folder_path, SAVE_NAME=SAVE_NAME):
    files = os.listdir(folder_path)
    files = [os.path.splitext(file)[0] for file in files if not file.startswith('.')]
    foldername = SAVE_NAME
    
    num = 0
    if len(files) > 0:
        for file in files:
            num = max(num, int(file.split('_')[-1]) + 1)

    foldername += str(num)
    folderpath = os.path.join(folder_path, foldername)
    model.save(folderpath)
    return folderpath


def get_latest_model(folder_path, SAVE_NAME=SAVE_NAME):
    files = os.listdir(folder_path)
    files = [os.path.splitext(file)[0] for file in files if not file.startswith('.')]
    foldername = SAVE_NAME

    num = 0
    if len(files) > 0:
        for file in files:
            num = max(num, int(file.split('_')[-1]))

    foldername += str(num)
    folderpath = os.path.join(folder_path, foldername)
    return folderpath


idx_to_labels = { k : v for k, v in enumerate(CATEGORIES)}
labels_to_idx = { k : v for v, k in enumerate(CATEGORIES)}
BERT_DIM = 384
NUM_LABELS = len(CATEGORIES)

# Load model + BERT encoder on start
MODEL_PATH = get_latest_model(SAVE_FOLDER_PATH)
print(f"Model loaded from: {MODEL_PATH}")

BERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
MLP_MODEL = tf.keras.models.load_model(MODEL_PATH)


def predict(model, bert, transactions):
    # Convert transaction descriptions into embeddings
    embeddings = bert.encode(transactions)

    predictions = model.predict(embeddings)
    maxProbs = predictions.max(-1)
    predicted_indices = np.argmax(predictions, axis=-1)
    predicted_categories = [idx_to_labels[idx] for idx in predicted_indices]

    return predicted_categories, maxProbs


# Load rules
rules = pd.read_csv(RULES_PATH)
case_sensitive_rules = pd.read_csv(CASE_SENSITIVE_RULES_PATH)

# Predict request schema
class Transaction(BaseModel):
    description: str

class TransactionsRequest(BaseModel):
    transactions: List[Transaction]

# Train request schema
class TrainTransactions(BaseModel):
    description: str
    category: str

class TrainTransactionsRequest(BaseModel):
    train_transactions: List[TrainTransactions]

# Response schema
class Prediction(BaseModel):
    description: str
    predicted_category: str
    confidence: float

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=List[Prediction])
def predict_path(request: TransactionsRequest):

    # Extract memos
    descriptions = [format_memo(t.description, False) for t in request.transactions]

    # Predict
    preds, probabilities = predict(MLP_MODEL, BERT_MODEL, descriptions)

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
        Prediction(description=d, predicted_category=p, confidence=c)
        for d, p, c in zip(descriptions, predicted_categories, probabilities)
    ]

    return results


@app.post("/train")
def train_route(request: TrainTransactionsRequest):
    # Use globally declared model
    global MLP_MODEL

    # Extract memos
    descriptions = [format_memo(t.description, False) for t in request.train_transactions]
    categories = [t.category for t in request.train_transactions]
    embeddings = BERT_MODEL.encode(descriptions)
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

# @app.post("/predict", response_model=List[Prediction])
# def predict(request: TransactionsRequest):

#     # Extract memos
#     descriptions = [format_memo(t.description, False) for t in request.transactions]

#     # Encode with BERT
#     embeddings = BERT_MODEL.encode(descriptions)

#     # Predict
#     preds = model.predict(embeddings)

#     predicted_categories = []
#     for i, d in enumerate(descriptions):
        
#         matches = case_sensitive_rules[
#             case_sensitive_rules["company_name"].apply(lambda x: x in d)
#         ]
#         if not matches.empty:
#             predicted_categories.append(matches.iloc[0]["category"])
#             continue

#         matches = rules[
#             rules["company_name"].apply(lambda x: x in d)
#         ]
#         if not matches.empty:
#             predicted_categories.append(matches.iloc[0]["category"])
#             continue

#         predicted_categories.append(preds[i])
        
#     # Build response
#     results = [
#         Prediction(description=d, predicted_category=p)
#         for d, p in zip(descriptions, predicted_categories)
#     ]

#     return results