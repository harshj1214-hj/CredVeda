import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from features import compute_financial_ratios

def train_and_export():
    os.makedirs('artifacts', exist_ok=True)
    df = pd.read_csv("data/Credit_Score_Classification_Updated.csv")

    if "Customer Code" in df.columns:
        df = df.drop(columns=["Customer Code"])

    label_encoders = {}
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if "Credit Score" in cat_cols:
        cat_cols.remove("Credit Score")

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    df = compute_financial_ratios(df)

    drop_cols = ["Credit Score", "Payment History", "Debt"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df["Credit Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'artifacts/model.joblib')
    joblib.dump(label_encoders, 'artifacts/encoders.joblib')
    joblib.dump(list(X.columns), 'artifacts/feature_names.joblib')
    joblib.dump(X_train.sample(min(100, len(X_train)), random_state=42), 'artifacts/background.joblib')
    print("SUCCESS: Model and artifacts saved in artifacts/")

if __name__ == "__main__":
    train_and_export()