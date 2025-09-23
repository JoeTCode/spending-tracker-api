import pandas as pd
import numpy as np
from utils.evaluate import accuracy
from config import BARCLAYS_CSV_24_25, BARCLAYS_CSV_21_24

train = pd.read_csv(BARCLAYS_CSV_21_24)
test = pd.read_csv(BARCLAYS_CSV_24_25)

# Load GPT predictions
gpt = pd.read_csv("C:/Users/Joe/Code/transaction-categoriser/gpt_predictions.csv")

# Add actual labels from test DataFrame
gpt['Actual'] = test['Category']

# Compare predicted vs actual
mismatches = gpt[gpt['Category'] != gpt['Actual']]

# Print mismatched rows
print(mismatches.head(50))