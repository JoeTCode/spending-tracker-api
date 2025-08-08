import os
import joblib

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