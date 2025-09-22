from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from utils.barclays.format_csv import format_csv
import pandas as pd
from utils.model_io import save_model, get_latest_model
from config import BARCLAYS_CSV_21_24, BARCLAYS_CSV_24_25, SAVE_SVM_PATH
import time
import os
from utils.file_io import save_predictions
from utils.evaluate import accuracy, category_distribution

script_dir = os.path.dirname(__file__)  # folder where current script lives

TRAIN, TRAIN_SAVE = True, False
PREDICT, PREDICT_SAVE = True, False
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
        # Save classifier
        return save_model(clf, os.path.join(script_dir, SAVE_SVM_PATH))

    return clf

def predict(data, clf=None):
     # Load BERT model
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text descriptions into embeddings
    X = bert.encode(data)

    if not clf:
        model_path = os.path.join(script_dir, SAVE_SVM_PATH)
        clf = joblib.load(get_latest_model(model_path))
        if not clf:
            print(f"Could not find saved model at: {model_path}. Instantiating new classifier.")
            clf = SVC()
    
    return clf.predict(X)


clf = None
if TRAIN:
    csv_21_24 = pd.read_csv(BARCLAYS_CSV_21_24)
    labels = csv_21_24['Category']
    data = csv_21_24['Memo']
    
    if TRAIN_SAVE:
        filepath = train(labels, data, TRAIN_SAVE)
        print(filepath)
    else:
        clf = train(labels, data, TRAIN_SAVE)

    
if PREDICT:
    start = time.time()
    uncategorised_df = format_csv(BARCLAYS_CSV_24_25).drop(columns=['Category'])
    predictions = None

    if clf:
        predictions = predict(uncategorised_df['Memo'], clf)
    else:
        predictions = predict(uncategorised_df['Memo'])
        
    uncategorised_df['Category'] = predictions

    if PREDICT_SAVE:
        path = save_predictions(uncategorised_df, os.path.join(script_dir, 'predictions'), SAVE_PREDICTIONS_MODEL_DIR)
        print("Predictions CSV saved at:", path)
    else:
        print(uncategorised_df.head())

    print(f"Accuracy score: {accuracy(uncategorised_df, format_csv(BARCLAYS_CSV_24_25))*100:.2f}%")
    print(f'SVM prediction completed in: {time.time() - start:.2f} seconds on {len(predictions)} transaction(s).')