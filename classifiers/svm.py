from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from utils.barclays.format_csv import format_csv
import pandas as pd
from utils.model_io import save_model, get_latest_model
from config import BARCLAYS_CSV_21_24, BARCLAYS_CSV_24_25, BARCLAYS_CAT_COL, BARCLAYS_DESC_COL, RULES_PATH, CASE_SENSITIVE_RULES_PATH, CATEGORIES, SAVE_LOGREG_PATH
import time
import os
from utils.file_io import save_predictions
from utils.evaluate import accuracy, category_distribution

script_dir = os.path.dirname(__file__)  # folder where current script lives

TRAIN, TRAIN_SAVE = False, False
PREDICT, PREDICT_SAVE = False, False
RANDOM_STATE = 42
SAVE_PREDICTIONS_MODEL_DIR = 'svm'

def train(Y, data, save=False, random_state=RANDOM_STATE):
    # Load BERT model
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text descriptions into embeddings
    X = bert.encode(data)

    # Split and train logistic regression
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

    clf = SVC()
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    if save:
        # Save classifier and BERT model separately
        return save_model(clf, os.path.join(script_dir, SAVE_LOGREG_PATH))


def predict(data):
     # Load BERT model
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text descriptions into embeddings
    X = bert.encode(data)

    clf = joblib.load(get_latest_model(os.path.join(script_dir, SAVE_LOGREG_PATH)))
    if not clf:
        clf = SVC()

    return clf.predict(X)

if TRAIN:
    csv_21_24 = pd.read_csv(BARCLAYS_CSV_21_24)
    labels = csv_21_24['Category']
    data = csv_21_24['Memo']
    
    if TRAIN_SAVE:
        filepath = train(labels, data, TRAIN_SAVE)
        print(filepath)
    else: train(labels, data, TRAIN_SAVE)

    
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
    print(f'Logistic regression prediction completed in: {time.time() - start:.2f} seconds on {len(predictions)} transaction(s).')

print(category_distribution(pd.read_csv(BARCLAYS_CSV_24_25)))