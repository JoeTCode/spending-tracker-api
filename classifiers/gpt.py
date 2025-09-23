from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from config import BARCLAYS_CSV_24_25, BARCLAYS_CSV_21_24
import pandas as pd
from utils.evaluate import accuracy
import time

train = pd.read_csv(BARCLAYS_CSV_21_24)
test = pd.read_csv(BARCLAYS_CSV_24_25)

# Load .env file
load_dotenv()

client = OpenAI()

# Pydantic model for a classified transaction
class ClassifiedTransaction(BaseModel):
    description: str
    category: str

# Pydantic model for multiple transactions
class ClassifiedTransactionList(BaseModel):
    transactions: List[ClassifiedTransaction]

# Define categories
CATEGORIES = [
    "Groceries", "Housing & Bills", "Finance & Fees", "Transport", "Income",
    "Shopping", "Eating Out", "Entertainment", "Health & Fitness",
    "Transfer", "Other / Misc"
]
    
# Transactions data
transactions = list(test["Memo"])

# Build system prompt
system_prompt = f"""
You are a financial transactions classifier. 
Classify each transaction into one of the following categories: {CATEGORIES}.
Return the output as a list of JSON objects with 'description' and 'category' fields, in the same order as the input transactions.
"""

# Build user prompt
user_prompt = "Here are the transactions:\n" + "\n".join(transactions)

start = time.time()
# Call OpenAI Responses API
response = client.responses.parse(
    model="gpt-5-nano",
    input=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    text_format=ClassifiedTransactionList,
)
print(f"Time taken for API call on {len(transactions)} transactions: {(time.time() - start):.4f} seconds")

# The output is a list of ClassifiedTransaction objects
classified_transactions: List[ClassifiedTransaction] = response.output_parsed.transactions

predictions = [pred.category for pred in classified_transactions]

# Pad predictions if length mismatch
if len(predictions) < len(test):
    missing = len(test) - len(predictions)
    predictions.extend(["Other / Misc"] * missing)

# Assign to DataFrame
predicted_df = test.assign(Category=predictions)

print(f"MLP accuracy: {accuracy(predicted_df, test) * 100:.2f}%")
predicted_df.to_csv('gpt_predictions.csv')
