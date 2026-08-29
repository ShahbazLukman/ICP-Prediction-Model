# 🎯 Behavior-Based Course Purchase Prediction & Lead Prioritization

An end-to-end machine learning project that predicts whether an education lead is likely to purchase a course within 30 days using behavioral and lead information.

> **Dataset note:** The included dataset is synthetic and was generated for learning/portfolio development. Do not describe it as real customer data. For a production project, replace it with properly licensed real lead/enrollment data.

## Business Problem

An education company receives many course enquiries. The sales team cannot contact every lead with the same priority.

This project answers:

1. Which leads are most likely to purchase within 30 days?
2. What is each lead's predicted purchase probability?
3. How can leads be ranked for sales follow-up?
4. What course was initially enquired about?
5. How can model performance be measured using both ML and business metrics?

## ML Problem

Primary target:

`purchased_within_30_days`

- `1` = purchased within 30 days
- `0` = did not purchase within 30 days

This is a **binary classification** problem.

## Models

The training script compares:

- Logistic Regression
- Random Forest
- XGBoost

It then performs randomized hyperparameter tuning for XGBoost using stratified 5-fold cross-validation.

Metrics:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Brier score
- Confusion matrix
- Top-K lift

## Project Structure

```text
course_lead_prediction_project/
├── data/
│   └── raw/
│       └── synthetic_course_leads.csv
├── notebooks/
│   └── 01_data_understanding_eda.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── models/
├── reports/
├── app/
│   └── app.py
├── requirements.txt
└── README.md
```

## Step-by-Step Setup

### Step 1 — Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2 — Install packages

```bash
pip install -r requirements.txt
```

### Step 3 — Explore the data

Open:

```text
notebooks/01_data_understanding_eda.ipynb
```

Run the cells from top to bottom.

### Step 4 — Train the models

From the project root:

```bash
python src/train.py
```

This creates:

```text
models/purchase_model.joblib
reports/model_comparison.csv
reports/final_metrics.json
reports/confusion_matrix.png
reports/threshold_analysis.csv
```

### Step 5 — Run the Streamlit app

```bash
streamlit run app/app.py
```

Then open the local URL shown by Streamlit.

## What the model learns

The model uses information that should be available before purchase, such as:

- course enquiry
- website visits
- course-page views
- pricing-page views
- brochure download
- email engagement
- WhatsApp responses
- calls answered
- follow-up count
- webinar/demo attendance
- video watches
- days since last activity
- basic lead information

## Important Data Leakage Rule

Do not use information that happens after the prediction point.

For example, `purchased_course` cannot be used to predict `purchased_within_30_days`.

In a real company, define a prediction timestamp and only use events known before that timestamp.

## How to Improve This Project

After the first version works:

1. Replace synthetic data with a properly licensed real dataset.
2. Add time-based features such as activities in the last 7/30 days.
3. Compare multiple models.
4. Tune the best model.
5. Calibrate probabilities.
6. Add SHAP explanations.
7. Test top-5%, top-10%, and top-20% lead lift.
8. Add a course-level purchase model if actual purchased-course labels are available.
9. Add model monitoring and data drift checks.
10. Deploy the Streamlit application.

## Resume Description

Use actual measured results after running the project. Example format:

> Built an end-to-end machine learning lead-scoring system to predict 30-day course purchase probability from lead engagement behavior; compared classification models using stratified cross-validation, optimized the best model, evaluated precision/recall and top-K lift, and deployed an interactive Streamlit application for individual and batch lead scoring.

Do not claim metrics until you have actually measured them.

## Ethical/Business Note

This project is for educational/portfolio purposes. In real deployment, avoid using sensitive attributes for decisions without a legitimate business/legal basis, audit performance across relevant groups, and communicate that predictions are probabilistic rather than guarantees.
