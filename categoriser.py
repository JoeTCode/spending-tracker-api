from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from utils.barclays.format_csv import format_csv

CATEGORIES = ["Groceries", "Housing & Bills", "Transport", "Income", "Shopping", "Eating Out", "Entertainment", "Health & Fitness", "Other / Misc"]
csv_24_25_filepath = './data/barclays/24-25/categorised-transaction-info-2024-25.csv'
csv_21_24_filepath = './data/barclays/21-24/uncategorised-transaction-info-2021-24.csv'
save_logreg_folderpath = './model-weights/logreg'

df = format_csv(csv_21_24_filepath)

def save_model(model, folder_path):
    files = os.listdir(folder_path)
    files = [os.path.splitext(file)[0] for file in files if not file.startswith('.')]
    filename = 'model_weights_'
    model_num = 0
    if len(files) > 0:
        for file in files:
            model_num = max(model_num, int(file.split('_')[-1]) + 1)
    filename += str(model_num)
    filepath = os.path.join(folder_path, filename + '.joblib')
    joblib.dump(model, filepath)
    return filepath

def get_latest_model(folder_path):
    files = os.listdir(folder_path)
    files = [os.path.splitext(file)[0] for file in files if not file.startswith('.')]
    filename = 'model_weights_'
    model_num = 0
    if len(files) > 0:
        for file in files:
            model_num = max(model_num, int(file.split('_')[-1]))
    filename += str(model_num)
    filepath = os.path.join(folder_path, filename + '.joblib')
    return filepath
    

def train_logreg_classifier(Y, data, save=False):
    # Load BERT model
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text descriptions into embeddings
    X = bert.encode(data)

    # Split and train logistic regression
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    if save:
        # Save classifier and BERT model separately
        return save_model(clf, save_logreg_folderpath)


def predict_logreg_classifier(data):
     # Load BERT model
    bert = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert text descriptions into embeddings
    X = bert.encode(data)

    clf = joblib.load(get_latest_model(save_logreg_folderpath))

    predictions = clf.predict(X)
    return data, predictions


# data, predictions = predict_logreg_classifier(format_csv(csv_21_24_filepath)['Memo'])
# for d, p in zip(data, predictions):
#     print(f'{d} \t {p}')


# labels = df['Category']
# memo = df['Memo']
# save = True
# if save:
#     filepath = train_logreg_classifier(labels, memo, save)
#     print(filepath)
# else: train_logreg_classifier(labels, memo, save)

def get_most_common_in_category(category, parent_path):
    category_to_foldername = {
        'eating out': 'eating-out', 'entertainment': 'entertainment', 'groceries': 'groceries',
        'housing & utilities': 'housing-&-utilities', 'shopping': 'shopping', 'transport': 'transport'
        }
    
    if category.lower() not in category_to_foldername.keys():
        print(f'Category ${category} is invalid')
        return -1
    
    category_folder_filepath = os.path.join(parent_path, category_to_foldername[category.lower()])
    return os.listdir(category_folder_filepath)

# def rule_based_filter(memo):

# uncategorised_df = format_csv(csv_21_24_filepath)['Memo']
# uncategorised_df['Memo'] = uncategorised_df['Memo'].apply(lambda memo: )

print(get_most_common_in_category('eating out', './data'))