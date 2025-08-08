import pandas as pd

def accuracy(df, targets, category_col='Category'):
    df = df[category_col].str.strip().str.lower()
    tgts = targets[category_col].str.strip().str.lower()
    mask = df == tgts
    return mask.mean()

def category_distribution(df, category_col='Category', description_col='Memo'):
    print('Number of uniques values in each category:')
    print(df.groupby(category_col)[description_col].nunique().sort_values())
    print('')

# logreg_df = pd.read_csv('predicted-no-rule-based.csv')
# targets = pd.read_csv(BARCLAYS_CSV_24_25)['Category']

# # log reg preds
# print(f"Logreg preds: {compare(logreg_df['Category'], targets):.2%}")

# # rule based log reg preds
# rules_logreg_preds = rule_based_categoriser(logreg_df)
# print(f"Rules-based logreg preds: {compare(rules_logreg_preds, targets):.2%}")

# # rule based preds
# rules_preds = rule_based_categoriser(logreg_df.drop(columns=['Category']))
# print(f"Rules-based preds: {compare(rules_preds, targets):.2%}")
# print('')

# # Show differences
# logreg_df['Rules_Category'] = rules_logreg_preds
# logreg_df['Target'] = targets
# difference = logreg_df[logreg_df['Rules_Category'].str.strip().str.lower() != logreg_df['Category'].str.strip().str.lower()]
# print(difference)