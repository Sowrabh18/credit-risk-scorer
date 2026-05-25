# ─────────────────────────────────────────────────────
# src/evaluate.py
#
# Model evaluation and SHAP explainability.
# Run after training to understand model behaviour.
# ─────────────────────────────────────────────────────

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml
from src.preprocess import preprocess

# ── Constants ───────────────────────────────────────────
MODEL_PATH = 'models/credit_risk_model.pkl'

# ── Load params ─────────────────────────────────────────
with open('params.yml', 'r') as f:
    params = yaml.safe_load(f)

# ── Load model and prepare data ─────────────────────────
def load_model_and_data():
    """
    Loads the trained model and recreates
    the exact same train/test split as training.

    Returns:
    - model: the trained Pipeline
    - X_test: test features (raw, before encoding)
    - y_test: test targets
    - X_test_transformed: test features after encoding
      (what the model actually sees)
    - feature_names: column names after OHE encoding
    """

    # Load trained model
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    # Get preprocessed data
    X, y = preprocess()

    # Recreate exact same split as training
    # Must use same random_state and test_size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['data']['test_size'],
        random_state=params['data']['random_state'],
        stratify=y
    )

    print(f"Test set: {len(X_test)} rows")

    # Get the transformed version of X_test
    # This is what the model actually sees after
    # OHE and StandardScaler have been applied
    preprocessor = model.named_steps['preprocessor']
    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names after OHE
    # OHE creates multiple columns from one
    # e.g. Sex becomes Sex_male and Sex_female
    ohe = preprocessor.named_transformers_['onehot']
    cat_feature_names = ohe.get_feature_names_out(
        input_features=['Sex', 'Housing',
                        'Saving accounts',
                        'Checking account',
                        'Purpose']
    ).tolist()

    numeric_feature_names = [
        'Age', 'Job', 'Credit amount', 'Duration',
        'saving_missing', 'checking_missing'
    ]

    feature_names = cat_feature_names + numeric_feature_names

    return model, X_test, y_test, X_test_transformed, feature_names

# ── Run SHAP analysis ────────────────────────────────────
def run_shap_analysis(model, X_test_transformed, feature_names):
    """
    Calculates SHAP values and generates three plots:
    1. Bar plot - overall feature importance
    2. Summary plot - feature impact distribution
    3. Waterfall plot - one specific prediction explained
    """

    print("\nCalculating SHAP values...")
    print("This may take a moment...")

    # ── Create SHAP explainer ─────────────────────────
    # We use the classifier step from inside the Pipeline
    # LinearExplainer is specifically for linear models
    # like Logistic Regression
    classifier = model.named_steps['classifier']

    explainer = shap.LinearExplainer(
        classifier,
        X_test_transformed,
        feature_perturbation='interventional'
    )

    # ── Calculate SHAP values ─────────────────────────
    shap_values = explainer.shap_values(X_test_transformed)

    print(f"SHAP values calculated for {len(X_test_transformed)} test samples")
    print(f"Shape of SHAP values: {shap_values.shape}")

    return explainer, shap_values

# ── Generate plots ───────────────────────────────────────
def generate_plots(explainer, shap_values,
                   X_test_transformed, feature_names):
    """
    Generates and saves three SHAP plots.
    """

    import os
    os.makedirs('models', exist_ok=True)

    # ── Plot 1: Bar plot ──────────────────────────────
    # Shows average absolute SHAP value per feature
    # Higher bar = more important feature overall
    print("\nGenerating bar plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names,
        plot_type='bar',
        show=False
    )
    plt.title('Feature Importance — Average SHAP Values')
    plt.tight_layout()
    plt.savefig('models/shap_bar_plot.png',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: models/shap_bar_plot.png")

    # ── Plot 2: Summary plot ──────────────────────────
    # Shows distribution of SHAP values for each feature
    # Color shows feature value (red=high, blue=low)
    # Position shows impact (right=bad risk, left=good risk)
    print("\nGenerating summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.title('SHAP Summary Plot — Feature Impact Distribution')
    plt.tight_layout()
    plt.savefig('models/shap_summary_plot.png',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: models/shap_summary_plot.png")

    # ── Plot 3: Waterfall plot ────────────────────────
    # Shows why the model made ONE specific prediction
    # We explain the first bad risk prediction in test set
    print("\nGenerating waterfall plot...")
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test_transformed[0],
            feature_names=feature_names
        ),
        show=False
    )
    plt.title('SHAP Waterfall — Single Prediction Explained')
    plt.tight_layout()
    plt.savefig('models/shap_waterfall_plot.png',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: models/shap_waterfall_plot.png")

# ── Print feature importance ─────────────────────────────
def print_feature_importance(shap_values, feature_names):
    """
    Prints a ranked list of features by importance.
    Importance = mean absolute SHAP value.
    """

    # Calculate mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame for easy sorting and display
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)

    print("\n" + "="*50)
    print("FEATURE IMPORTANCE RANKING")
    print("="*50)
    print(f"{'Rank':<6} {'Feature':<35} {'Importance':>10}")
    print("-"*55)
    for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
        print(f"{rank:<6} {row['feature']:<35} {row['importance']:>10.4f}")

    return importance_df

# ── Main function ────────────────────────────────────────
def main():
    """
    Full SHAP analysis pipeline:
    1. Load model and data
    2. Calculate SHAP values
    3. Generate plots
    4. Print feature importance
    """

    print("="*50)
    print("SHAP Explainability Analysis")
    print("="*50)

    # Load everything
    model, X_test, y_test, X_test_transformed, feature_names = \
        load_model_and_data()

    # Calculate SHAP values
    explainer, shap_values = run_shap_analysis(
        model, X_test_transformed, feature_names
    )

    # Generate plots
    generate_plots(
        explainer, shap_values,
        X_test_transformed, feature_names
    )

    # Print importance ranking
    importance_df = print_feature_importance(
        shap_values, feature_names
    )

    print("\n" + "="*50)
    print("SHAP analysis complete")
    print("Plots saved to models/ folder")
    print("="*50)

    return importance_df


# ── Entry point ──────────────────────────────────────────
if __name__ == '__main__':
    main()