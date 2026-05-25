# ─────────────────────────────────────────────────────
# src/train.py
#
# Trains credit risk models with hyperparameter tuning.
# Hyperparameters come from params.yml.
# Every experiment tracked with MLflow.
# ─────────────────────────────────────────────────────

import yaml
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from src.preprocess import preprocess
# ── Constants ───────────────────────────────────────────
MODEL_PATH = 'models/credit_risk_model.pkl'
MLFLOW_EXPERIMENT_NAME = 'credit-risk-scorer'

# ── Load hyperparameters from params.yml ────────────────
with open('params.yml', 'r') as f:
    params = yaml.safe_load(f)

# ── Column definitions ──────────────────────────────────
# These must match exactly what preprocess.py returns
CATEGORICAL_COLUMNS = [
    'Sex',
    'Housing',
    'Saving accounts',
    'Checking account',
    'Purpose'
]

NUMERIC_COLUMNS = [
    'Age',
    'Job',
    'Credit amount',
    'Duration',
    'saving_missing',
    'checking_missing'
]

# ── Build column transformer ────────────────────────────
def build_preprocessor():
    """
    Creates a ColumnTransformer that applies:
    - OneHotEncoder to categorical columns
    - StandardScaler to numeric columns

    StandardScaler fixes the convergence warning
    by putting all numeric features on the same scale.
    Fitted on training data only — no leakage.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'onehot',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ),
                CATEGORICAL_COLUMNS
            ),
            (
                'numeric',
                StandardScaler(),
                NUMERIC_COLUMNS
            )
        ]
    )

    return preprocessor

# ── Split data ──────────────────────────────────────────
def split_data(X, y):
    """
    Splits features and target into
    training and test sets.

    Settings come from params.yml:
    - test_size: fraction for test set
    - random_state: for reproducibility
    - stratify: maintains class ratio in both sets
    """

    test_size = params['data']['test_size']
    random_state = params['data']['random_state']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Training set: {len(X_train)} rows")
    print(f"Test set:     {len(X_test)} rows")
    print(f"Train default rate: {y_train.mean():.1%}")
    print(f"Test default rate:  {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test

# ── Train and evaluate one model ────────────────────────
def train_model(X_train, X_test, y_train, y_test,
                model_name, model, param_grid):
    """
    Trains one model with hyperparameter tuning.
    Logs everything to MLflow.

    Parameters:
    - model_name: string label for MLflow
    - model: sklearn model object
    - param_grid: hyperparameters to search
    """

    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")

    # ── Start MLflow run ───────────────────────────────
    # Everything inside this block gets recorded
    # as one experiment run in MLflow
    with mlflow.start_run(run_name=model_name):

        # ── Log params.yml settings to MLflow ─────────
        # This links the MLflow run to the exact
        # hyperparameter settings that produced it
        mlflow.log_params(params['data'])
        mlflow.log_param('model_name', model_name)

        # ── Build pipeline ─────────────────────────────
        # Step 1: ColumnTransformer encodes features
        # Step 2: model learns from encoded features
        # One object handles both steps together
        preprocessor = build_preprocessor()

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # ── Hyperparameter search ──────────────────────
        cv_folds = params['evaluation']['cv_folds']
        primary_metric = params['evaluation']['primary_metric']

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_folds,
            scoring=primary_metric,
            n_jobs=-1,
            verbose=1
        )

        print("Running hyperparameter search...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")

        # ── Evaluate on test set ───────────────────────
        y_pred = best_model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)

        print(f"\nResults:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"\nDetailed report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['good', 'bad']
        ))

        # ── Log everything to MLflow ───────────────────
        # ── Log metrics and params to MLflow ──────────
        mlflow.log_param('best_params', str(best_params))
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('precision", prec)
        mlflow.log_metric('recall', rec)
        mlflow.log_metric('f1_score', f1)
        mlflow.sklearn.log_model(
            best_model,
            model_name.lower().replace(' ', '_')
        )

        # ── Run SHAP and log plots to MLflow ──────────
        # Only run SHAP for the best model — Logistic
        # Regression. Skip for other models to save time.
        try:
            import shap
            import matplotlib.pyplot as plt
            import os

            os.makedirs('reports', exist_ok=True)

            # Get transformed test data
            preprocessor_step = best_model.named_steps['preprocessor']
            X_test_transformed = preprocessor_step.transform(X_test)

            # Get classifier from pipeline
            classifier_step = best_model.named_steps['classifier']

            # Only LinearExplainer works for Logistic Regression
            # Skip SHAP for Random Forest here since we use
            # TreeExplainer for that — we'll add it later
            if 'LogisticRegression' in str(type(classifier_step)):

                explainer = shap.LinearExplainer(
                    classifier_step,
                    X_test_transformed
                )
                shap_values = explainer.shap_values(X_test_transformed)

                # Get feature names
                ohe = preprocessor_step.named_transformers_['onehot']
                from src.preprocess import CATEGORICAL_COLUMNS
                cat_names = ohe.get_feature_names_out(
                    input_features=CATEGORICAL_COLUMNS
                ).tolist()
                from src.train import NUMERIC_COLUMNS
                feature_names = cat_names + NUMERIC_COLUMNS

                # Bar plot
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
                plt.savefig('reports/shap_bar_plot.png',
                            bbox_inches='tight', dpi=150)
                plt.close()

                # Summary plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    X_test_transformed,
                    feature_names=feature_names,
                    show=False
                )
                plt.title('SHAP Summary — Feature Impact Distribution')
                plt.tight_layout()
                plt.savefig('reports/shap_summary_plot.png',
                            bbox_inches='tight', dpi=150)
                plt.close()

                # Waterfall plot
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
                plt.savefig('reports/shap_waterfall_plot.png',
                            bbox_inches='tight', dpi=150)
                plt.close()

                # Log all three plots to MLflow
                mlflow.log_artifact('reports/shap_bar_plot.png')
                mlflow.log_artifact('reports/shap_summary_plot.png')
                mlflow.log_artifact('reports/shap_waterfall_plot.png')

                print("  SHAP plots logged to MLflow")

        except Exception as e:
            print(f"  SHAP analysis skipped: {e}")

        return {
            'model_name': model_name,
            'model': best_model,
            'f1_score': f1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'params': best_params
        }
    

# ── Main function ────────────────────────────────────────
def main():
    """
    Full training pipeline:
    1. Load and preprocess data
    2. Split into train/test
    3. Train multiple models
    4. Compare results
    5. Save best model
    """

    # ── Set up MLflow ──────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── Get data ───────────────────────────────────────
    print("Loading and preprocessing data...")
    X, y = preprocess()

    # ── Split data ─────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Define models and parameter grids ─────────────
    # Hyperparameters come from params.yml
    # Parameter names use double underscore __ because
    # they're inside a Pipeline
    # 'preprocessor__onehot__...' means the onehot step
    # inside the preprocessor step inside the pipeline
    # 'classifier__C' means the C parameter of
    # the classifier step inside the pipeline
    experiments = [
        {
            'name': 'Logistic Regression',
            'model': LogisticRegression(
                max_iter=params['logistic_regression']['max_iter'],
                class_weight=params['logistic_regression']['class_weight'],
                random_state=params['data']['random_state']
            ),
            'param_grid': {
                'classifier__C': params['logistic_regression']['C']
            }
        },
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(
                class_weight=params['random_forest']['class_weight'],
                random_state=params['data']['random_state'],
                n_jobs=-1
            ),
            'param_grid': {
                'classifier__n_estimators': params['random_forest']['n_estimators'],
                'classifier__max_depth': params['random_forest']['max_depth']
            }
        }
    ]

    # ── Train all models ───────────────────────────────
    results = []
    for exp in experiments:
        result = train_model(
            X_train, X_test, y_train, y_test,
            exp['name'], exp['model'], exp['param_grid']
        )
        results.append(result)

    # ── Compare results ────────────────────────────────
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    print(f"{'Model':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Accuracy':>10}")
    print("-"*65)
    for r in results:
        print(
            f"{r['model_name']:<25}"
            f"{r['f1_score']:>8.4f}"
            f"{r['precision']:>10.4f}"
            f"{r['recall']:>8.4f}"
            f"{r['accuracy']:>10.4f}"
        )

    # ── Pick best model ────────────────────────────────
    best = max(results, key=lambda x: x['f1_score'])
    print(f"\nBest model: {best['model_name']}")
    print(f"Best F1:    {best['f1_score']:.4f}")

    # ── Save best model ────────────────────────────────
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(best['model'], MODEL_PATH)
    print(f"Best model saved to: {MODEL_PATH}")

    return best


# ── Entry point ──────────────────────────────────────────
if __name__ == '__main__':
    main()