# ─────────────────────────────────────────────────────
# src/predict.py
#
# Loads the trained model and exposes a predict function.
# Returns prediction, probability, and SHAP explanation.
# Used by main.py (FastAPI) to serve predictions.
# ─────────────────────────────────────────────────────

import joblib
import shap
import numpy as np
import pandas as pd
from src.preprocess import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# ── Load model once at startup ──────────────────────────
# Loaded when file is first imported — not on every request
# Loading takes time so we do it once and reuse
MODEL_PATH = 'models/credit_risk_model.pkl'
model = joblib.load(MODEL_PATH)

# ── Get preprocessor and classifier from Pipeline ───────
# We need these separately for SHAP
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']

# ── Get feature names after OHE ─────────────────────────
# OHE expands Sex into Sex_male and Sex_female etc.
# We need these names for SHAP explanation labels
ohe = preprocessor.named_transformers_['onehot']
cat_feature_names = ohe.get_feature_names_out(
    input_features=CATEGORICAL_COLUMNS
).tolist()
feature_names = cat_feature_names + NUMERIC_COLUMNS + ['saving_missing', 'checking_missing']

# Create a small background dataset for the masker
# We use the training data statistics from the scaler
# to create a representative background
explainer = shap.LinearExplainer(
    classifier,
    masker=shap.maskers.Independent(
        np.zeros((1, len(feature_names)))
    )
)

# ── Prediction function ─────────────────────────────────
def predict(application: dict) -> dict:
    """
    Takes a loan application as a dictionary.
    Returns prediction, probability, and SHAP explanation.

    Input example:
    {
        'Age': 25,
        'Sex': 'male',
        'Job': 2,
        'Housing': 'own',
        'Saving accounts': 'little',
        'Checking account': 'moderate',
        'Credit amount': 5000,
        'Duration': 24,
        'Purpose': 'car'
    }

    Saving accounts and Checking account are optional.
    If not provided they are treated as missing.
    """

    # ── Step 1: Handle missing values ───────────────────
    # Create indicator columns exactly like preprocessing
    # 1 means was missing, 0 means was provided
    saving_missing = 1 if application.get(
        'Saving accounts') is None else 0
    checking_missing = 1 if application.get(
        'Checking account') is None else 0

    # Fill missing with unknown
    # exactly like preprocessing does
    if application.get('Saving accounts') is None:
        application['Saving accounts'] = 'unknown'
    if application.get('Checking account') is None:
        application['Checking account'] = 'unknown'

    # ── Step 2: Build feature DataFrame ─────────────────
    # Must have exact same columns in exact same order
    # as what the Pipeline was trained on
    feature_row = {
        'Age': application['Age'],
        'Job': application['Job'],
        'Credit amount': application['Credit amount'],
        'Duration': application['Duration'],
        'Sex': application['Sex'],
        'Housing': application['Housing'],
        'Saving accounts': application['Saving accounts'],
        'Checking account': application['Checking account'],
        'Purpose': application['Purpose'],
        'saving_missing': saving_missing,
        'checking_missing': checking_missing
    }

    # Convert to DataFrame
    # Pipeline expects a DataFrame not a dictionary
    # [feature_row] wraps dict in a list — one row
    df_input = pd.DataFrame([feature_row])

    # ── Step 3: Get prediction ───────────────────────────
    # Pipeline handles OHE and scaling automatically
    # predict returns 0 (good) or 1 (bad)
    # predict_proba returns probability for each class
    prediction_encoded = model.predict(df_input)[0]
    probability_bad = model.predict_proba(df_input)[0][1]

    # Convert number back to readable label
    prediction_label = 'bad' if prediction_encoded == 1 else 'good'

    # Convert probability to percentage with 1 decimal
    risk_score = round(float(probability_bad) * 100, 1)

    # ── Step 4: Get SHAP explanation ─────────────────────
    # Transform input exactly as model does
    # SHAP needs the transformed version not raw input
    X_transformed = preprocessor.transform(df_input)

    # Calculate SHAP values using explainer created at startup
    # shap_values[0] because we have one row only
    shap_values = explainer.shap_values(X_transformed)[0]

    # ── Step 5: Get top 5 factors ────────────────────────
    # Pair each feature name with its SHAP value
    feature_impacts = list(zip(feature_names, shap_values))

    # Sort by absolute SHAP value — highest impact first
    # abs() because we care about magnitude not direction
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

    # Take top 5 only
    top_5 = feature_impacts[:5]

    # Build readable explanation for each factor
    top_factors = []
    for feature, shap_val in top_5:
        top_factors.append({
            'feature': feature,
            'impact': round(float(shap_val), 4),
            'direction': 'increases risk' if shap_val > 0 else 'decreases risk'
        })

    # ── Step 6: Return result ────────────────────────────
    return {
        'prediction': prediction_label,
        'probability': round(float(probability_bad), 4),
        'risk_score': risk_score,
        'top_factors': top_factors
    }


# ── Entry point ──────────────────────────────────────────
# Test with a sample application when run directly
if __name__ == '__main__':
    sample = {
        'Age': 25,
        'Sex': 'male',
        'Job': 2,
        'Housing': 'own',
        'Saving accounts': 'little',
        'Checking account': 'moderate',
        'Credit amount': 5000,
        'Duration': 24,
        'Purpose': 'car'
    }

    result = predict(sample)

    print("\nSample prediction:")
    print(f"  Prediction:  {result['prediction']}")
    print(f"  Risk score:  {result['risk_score']}%")
    print(f"  Probability: {result['probability']}")
    print(f"\nTop factors:")
    for factor in result['top_factors']:
        print(
            f"  {factor['feature']:<35}"
            f"  {factor['impact']:>8.4f}"
            f"  ({factor['direction']})"
        )