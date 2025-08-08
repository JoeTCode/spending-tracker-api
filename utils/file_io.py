import os
import pandas as pd

def save_predictions(predictions, save_root_directory_path, model_sub_directory):
    folder_path = os.path.join(save_root_directory_path, model_sub_directory)
    files = os.listdir(folder_path)
    files = [os.path.splitext(file)[0] for file in files if not file.startswith('.')]
    filename = 'predictions_'
    csv_num = 0

    if len(files) > 0:
        for file in files:
            csv_num = max(csv_num, int(file.split('_')[-1]) + 1)
            
    filename += str(csv_num)
    filepath = os.path.join(folder_path, filename + '.csv')
    predictions.to_csv(filepath, index=False)
    return filepath