# ⚡ CredVeda AI

**An explainable credit underwriting and recourse tool.**
Enter a borrower's income and debt, and CredVeda predicts a credit risk tier, shows *why* the model decided that, and tells the borrower *what to change* to get a better rating.

🌐 **Live app:** https://www.credveda.jainovation.xyz/
📦 **Source:** https://github.com/harshj1214-hj/CredVeda

---

## 💡 The problem

Most automated credit systems are black boxes. A borrower gets "rejected" with no reason and no next step. Lenders, in turn, have no easy way to show the reasoning behind a decision.

## 🎯 What CredVeda does

| What you see | What it means |
|---|---|
| **Risk tier + confidence** | The model's predicted tier (Prime / Moderate / Subprime) and how sure it is. |
| **Tier probabilities** | The chance the model gives to every tier, not just the winner. |
| **SHAP attribution** | A bar chart showing which inputs pushed the result up or down. |
| **Actionable recourse** | Plain-English advice, such as the debt prepayment needed to reach a better tier. |
| **Borrowing capacity** | Estimated maximum monthly EMI and 5-year loan size. |
| **Stress test** | See how the rating holds up if income drops or debt rises. |
| **JSON export** | Download the inputs, prediction, SHAP values and recourse as one file. |

## ✨ Features

- **4 currencies:** INR (₹), USD ($), EUR (€) and GBP (£), with amounts formatted for each (e.g. lakh/crore for INR).
- **Live indicators:** Debt-to-Income (DTI), free cash (disposable income), and a colour-coded DTI bar (green under 30%, amber under 45%, red above).
- **Visuals:** a credit score gauge, a 5-pillar radar (DTI Safety, Earning Scale, Leverage, Age Maturity, Surplus), and probability bars.
- **Stress test:** apply an income shock (−30% to +30%) and a liability spike (0% to +100%) and watch the result change.
- **Borrowing capacity:** uses the common **50% FOIR** rule (total fixed obligations capped at 50% of income) for a 5-year personal loan at 10.5% p.a.

## 🧠 How it works

```text
Applicant inputs (income, debt, age, demographics)
        │
        ▼
Feature engineering  ->  DTI = Debt / Income
(features.py)            Disposable Income = Income − Debt
        │
        ▼
Random Forest classifier  ->  predicted tier + class probabilities
(train.py)
        │
        ├──────────────────────────────┐
        ▼                              ▼
SHAP TreeExplainer              Counterfactual recourse
(which features mattered)       (smallest debt reduction that
(explain.py)                     moves the applicant to a better tier)
        │                              │
        └──────────────┬───────────────┘
                       ▼
              Streamlit dashboard (app.py)
```

**Step by step**

1. **Features.** `features.py` adds two ratios to the data: `DTI` (debt ÷ income) and `Disposable_Income` (income − debt).
2. **Training.** `train.py` encodes text columns, builds the features, splits the data 80/20 and trains a Random Forest (100 trees). It saves the model, encoders, feature list and a 100-row background sample to `artifacts/`.
3. **Prediction.** `app.py` loads those files and predicts a tier for the applicant on screen.
4. **Explanation.** `explain.py` uses SHAP to compute how much each feature pushed the prediction toward or away from the predicted tier.
5. **Recourse.** `explain.py` also raises the applicant's disposable income in small steps (i.e. pays off debt) and re-runs the model after each step. It stops at the first step where the tier improves and reports that amount. If no amount works, it suggests adding a co-borrower. Applicants already in the best tier are told to keep things as they are.

## 📁 Project structure

```text
CredVeda/
├── app.py                 # Streamlit dashboard (UI, charts, stress test, export)
├── requirements.txt
├── data/
│   └── Credit_Score_Classification_Updated.csv   # training data (add your own copy)
├── artifacts/             # created by training: model, encoders, features, background
└── src/
    ├── features.py        # DTI and disposable income calculation
    ├── train.py           # trains and saves the model
    └── explain.py         # SHAP contributions + counterfactual recourse
```

## 🚀 Run it locally

**1. Install**

```bash
git clone https://github.com/harshj1214-hj/CredVeda.git
cd CredVeda
pip install -r requirements.txt
```

The app needs `streamlit`, `pandas`, `numpy`, `scikit-learn`, `shap`, `plotly` and `joblib`.

**2. Train the model** (run from the project root, with the CSV in `data/`)

```bash
python src/train.py
```

You should see `SUCCESS: Model and artifacts saved in artifacts/`.

**3. Start the app**

```bash
streamlit run app.py
```

If you skip step 2, the app shows a "Model artifacts missing" message.

## 🛠️ Tech stack

Python · scikit-learn (Random Forest) · SHAP · pandas / NumPy · Streamlit · Plotly

## ⚠️ Good to know

- This is a **demo / learning project**, not a production lending system. Do not use it for real credit decisions.
- The model is trained on a public-style credit dataset and uses age, gender, education, marital status, home ownership and number of children alongside income and debt. Real lenders are legally restricted from using some of these (for example under the US Equal Credit Opportunity Act), so a production system would need a proper fairness review.
- The credit score on the gauge is a fixed representative value for each tier (Prime, Moderate, Subprime), not a separately calculated score.
- Borrowing capacity uses fixed assumptions (50% FOIR, 5 years, 10.5% interest).
- The recourse only looks at reducing debt. It does not model other changes such as payment history.

## 🗺️ Ideas for next steps

- Add a fairness check across demographic groups.
- Let users choose their own loan term and interest rate.
- Add more recourse options beyond debt reduction.

---
