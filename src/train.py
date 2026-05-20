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
        mlflow.log_param('best_params', str(best_params))
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('precision', prec)
        mlflow.log_metric('recall', rec)
        mlflow.log_metric('f1_score', f1)
        mlflow.sklearn.log_model(
            best_model,
            model_name.lower().replace(' ', '_')
        )

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