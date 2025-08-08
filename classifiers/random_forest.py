from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from utils.barclays.format_csv import format_csv
import pandas as pd
from utils.model_io import save_model, get_latest_model
from config import BARCLAYS_CSV_21_24, BARCLAYS_CSV_24_25, BARCLAYS_CAT_COL, BARCLAYS_DESC_COL, RULES_PATH, CASE_SENSITIVE_RULES_PATH, CATEGORIES, SAVE_LOGREG_PATH, SAVE_RANDOM_FOREST_PATH
import time
import os
from utils.file_io import save_predictions
from utils.evaluate import accuracy

script_dir = os.path.dirname(__file__)  # path of current script

TRAIN, TRAIN_SAVE = True, True
PREDICT, PREDICT_SAVE = True, False
RANDOM_STATE = 42
RF_PARAMS = {
    "n_estimators": 500,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
}
SAVE_PREDICTIONS_MODEL_DIR = 'random_forest'

def train(Y, data, save=False, *, random_state, **kwargs):

    # Load BERT model
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text descriptions into embeddings
    X = bert.encode(data)

    # Split and train logistic regression
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    clf = RandomForestClassifier(**kwargs)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    if save:
        # Save classifier and BERT model separately
        return save_model(clf, os.path.join(script_dir, SAVE_RANDOM_FOREST_PATH))

def predict(data, max_depth=2, random_state=42):
     # Load BERT model
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text descriptions into embeddings
    X = bert.encode(data)
    
    clf = joblib.load(get_latest_model(os.path.join(script_dir, SAVE_RANDOM_FOREST_PATH)))
    if not clf:
        clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)

    return clf.predict(X)

if TRAIN:
    csv_21_24 = pd.read_csv(BARCLAYS_CSV_21_24)
    labels = csv_21_24['Category']
    data = csv_21_24['Memo']
    
    if TRAIN_SAVE:
        filepath = train(labels, data, TRAIN_SAVE, random_state=RANDOM_STATE, **RF_PARAMS)
        print(filepath)
    else: train(labels, data, TRAIN_SAVE, random_state=RANDOM_STATE, **RF_PARAMS)

    
if PREDICT:
    start = time.time()
    uncategorised_df = format_csv(BARCLAYS_CSV_24_25).drop(columns=['Category'])
    predictions = predict(uncategorised_df['Memo'])
    uncategorised_df['Category'] = predictions

    if PREDICT_SAVE:
        path = save_predictions(uncategorised_df, os.path.join(script_dir, 'predictions'), SAVE_PREDICTIONS_MODEL_DIR)
        print("Predictions CSV saved at:", path)
    else:
        print(uncategorised_df.head())

    print(f"Accuracy score: {accuracy(uncategorised_df, format_csv(BARCLAYS_CSV_24_25))*100:.2f}%")
    print(f"Logistic regression prediction completed in: {time.time() - start:.2f} seconds on {len(predictions)} transaction(s).")