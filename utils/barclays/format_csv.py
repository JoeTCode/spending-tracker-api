import pandas as pd
import re

def format_memo(memo):
        formatted_memo = str(memo).split('\t')[0].strip().lower()
        return re.sub(r"\s+", " ", formatted_memo).strip()

def format_csv(filepath):    
    df = pd.read_csv(filepath)
    df = df.iloc[:-1]
    formatted_df = df.drop(columns=['Number', 'Account'])
    formatted_df['Memo'] = formatted_df['Memo'].apply(lambda memo: format_memo(memo))
    return formatted_df

