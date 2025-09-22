from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from utils.barclays.format_csv import format_csv
import pandas as pd
from utils.model_io import save_model, get_latest_model
from config import BARCLAYS_CSV_21_24, BARCLAYS_CSV_24_25, BARCLAYS_CAT_COL, BARCLAYS_DESC_COL, BARCLAYS_PRICE_COL, RULES_PATH, CASE_SENSITIVE_RULES_PATH, CATEGORIES, SAVE_LOGREG_PATH, SAVE_RANDOM_FOREST_PATH
import time
import os
from utils.file_io import save_predictions
from utils.evaluate import accuracy
import numpy as np
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(__file__)  # path of current script

TRAIN, TRAIN_SAVE = True, False
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
    X = data

    # Load BERT model
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    if isinstance(data, pd.Series):
        # Convert text descriptions into embeddings
        X = bert.encode(data)

    elif isinstance(data, pd.DataFrame):
        if BARCLAYS_DESC_COL in data.columns and BARCLAYS_PRICE_COL in data.columns:
            # Convert text descriptions into embeddings
            X = bert.encode(data[BARCLAYS_DESC_COL])
            
            # Reshape prices into a 2d array with shape: (n_samples, 1)
            prices = np.array(data[BARCLAYS_PRICE_COL]).reshape((-1, 1))
            prices_scaled = StandardScaler().fit_transform(prices)

            # Combine both embedded text and scaled amounts into shape: (n_samples, X_dim + 1)
            X = np.hstack([X, prices_scaled])
        else:
            raise Exception(f"Unexpected column names: {data.columns}")
    
    else:
        raise TypeError(f"Provided {type(X)}. Only arguments of type pd.Series or pd.Dataframe are allowed.")

    # Split and train logistic regression
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    clf = RandomForestClassifier(**kwargs)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    if save:
        # Save classifier and return saved model path
        return save_model(clf, os.path.join(script_dir, SAVE_RANDOM_FOREST_PATH))
    
    else: return clf

def predict(data, clf=None, max_depth=2, random_state=42):
    X = data
    
    # Load BERT model
    bert = SentenceTransformer("all-MiniLM-L6-v2")
    
    if isinstance(data, pd.Series):
        # Convert text descriptions into embeddings
        X = bert.encode(data)

    elif isinstance(data, pd.DataFrame):
        if BARCLAYS_DESC_COL in data.columns and BARCLAYS_PRICE_COL in data.columns:
            # Convert text descriptions into embeddings
            X = bert.encode(data[BARCLAYS_DESC_COL])
            
            # Reshape prices into a 2d array with shape: (n_samples, 1)
            prices = np.array(data[BARCLAYS_PRICE_COL]).reshape((-1, 1))
            prices_scaled = StandardScaler().fit_transform(prices)

            # Combine both embedded text and scaled amounts into shape: (n_samples, X_dim + 1)
            X = np.hstack([X, prices_scaled])
        
        else:
            raise Exception(f"Unexpected column names: {data.columns}")
            
    else:
        raise TypeError(f"Provided {type(X)}. Only arguments of type pd.Series or pd.Dataframe are allowed.")
    
    if not clf:
        model_path = os.path.join(script_dir, SAVE_RANDOM_FOREST_PATH)
        clf = joblib.load(get_latest_model(model_path))
        if not clf:
            print(f"Could not find saved model at: {model_path}. Instantiating new classifier.")
            clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    
    return clf.predict(X)

clf = None
if TRAIN:
    csv_21_24 = pd.read_csv(BARCLAYS_CSV_21_24)
    labels = csv_21_24['Category']
    data = csv_21_24[['Amount', 'Memo']]
    
    if TRAIN_SAVE:
        filepath = train(labels, data, TRAIN_SAVE, random_state=RANDOM_STATE, **RF_PARAMS)
        print(filepath)
    else: 
        clf = train(labels, data, TRAIN_SAVE, random_state=RANDOM_STATE, **RF_PARAMS)

    
if PREDICT:
    start = time.time()
    uncategorised_df = format_csv(BARCLAYS_CSV_24_25).drop(columns=['Category'])
    predictions = None
    
    if clf:
        predictions = predict(uncategorised_df[['Amount', 'Memo']], clf)
    else: predictions = predict(uncategorised_df[['Amount', 'Memo']])
    uncategorised_df['Category'] = predictions

    if PREDICT_SAVE:
        path = save_predictions(uncategorised_df, os.path.join(script_dir, 'predictions'), SAVE_PREDICTIONS_MODEL_DIR)
        print("Predictions CSV saved at:", path)
    else:
        print(uncategorised_df.head())

    print(f"Accuracy score: {accuracy(uncategorised_df, format_csv(BARCLAYS_CSV_24_25))*100:.2f}%")
    print(f"Logistic regression prediction completed in: {time.time() - start:.2f} seconds on {len(predictions)} transaction(s).")