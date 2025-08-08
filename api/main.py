from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
from sentence_transformers import SentenceTransformer
from utils.barclays.format_csv import format_memo
import os
from config import RULES_PATH, CASE_SENSITIVE_RULES_PATH
import pandas as pd

script_dir = os.path.dirname(__file__)  # path of current script

# Load model + BERT encoder on start
MODEL_PATH = os.path.join(script_dir, "model_weight/model_weights_1.joblib")
BERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
model = joblib.load(MODEL_PATH)

# Load rules
rules = pd.read_csv(RULES_PATH)
case_sensitive_rules = pd.read_csv(CASE_SENSITIVE_RULES_PATH)

# Request schema
class Transaction(BaseModel):
    description: str
    amount: float

class TransactionsRequest(BaseModel):
    transactions: List[Transaction]

# Response schema
class Prediction(BaseModel):
    description: str
    predicted_category: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=List[Prediction])
def predict(request: TransactionsRequest):
    # Extract memos
    descriptions = [format_memo(t.description, False) for t in request.transactions]

    # Encode with BERT
    embeddings = BERT_MODEL.encode(descriptions)

    # Predict
    preds = model.predict(embeddings)

    rules_preds = []
    for i, d in enumerate(descriptions):
        skip = False
        
        matches = case_sensitive_rules[
            case_sensitive_rules["company_name"].apply(lambda x: x in d)
        ]
        if not matches.empty:
            rules_preds.append(matches.iloc[0]["category"])
            skip = True

        matches = rules[
            rules["company_name"].apply(lambda x: x in d)
        ]
        if not matches.empty:
            rules_preds.append(matches.iloc[0]["category"])
            skip = True

        if not skip:
            rules_preds.append(preds[i])
        
    # Build response
    results = [
        Prediction(description=d, predicted_category=p)
        for d, p in zip(descriptions, rules_preds)
    ]
    return results