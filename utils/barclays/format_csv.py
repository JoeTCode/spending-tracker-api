import pandas as pd
import re

def format_memo(memo, lower=True):
        formatted_memo = str(memo).split('\t')[0].strip()
        if lower:
              formatted_memo.lower()
        return re.sub(r"\s+", " ", formatted_memo).strip()

def format_csv(filepath):    
    df = pd.read_csv(filepath)
    formatted_df = df.drop(columns=['Number', 'Account'])
    formatted_df['Memo'] = formatted_df['Memo'].apply(lambda memo: format_memo(memo))
    return formatted_df

