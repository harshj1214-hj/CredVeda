import pandas as pd

def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Debt' in df.columns and 'Income' in df.columns:
        df['DTI'] = df['Debt'] / (df['Income'] + 1e-5)
        df['Disposable_Income'] = df['Income'] - df['Debt']
    else:
        df['DTI'] = 0.0
        df['Disposable_Income'] = df['Income']
    return df