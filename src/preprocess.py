# ─────────────────────────────────────────────────────
# src/preprocess.py
#
# Loads and cleans the German Credit dataset.
# Returns features (X) and target (y) ready for training.
# Based on findings from notebooks/eda.ipynb
# ─────────────────────────────────────────────────────

import pandas as pd
import numpy as np
# ── File paths ─────────────────────────────────────────
RAW_DATA_PATH = 'data/raw/german_credit_data.csv'

# ── Column definitions ─────────────────────────────────
# These are the columns we'll apply OHE to
# Defined here at the top so if columns ever change
# we update in one place only
CATEGORICAL_COLUMNS = [
    'Sex',
    'Housing',
    'Saving accounts',
    'Checking account',
    'Purpose'
]

# These are columns we keep as numbers unchanged
NUMERIC_COLUMNS = [
    'Age',
    'Job',
    'Credit amount',
    'Duration'
]

# This is our target variable
TARGET_COLUMN = 'Risk'

# ── Load data ───────────────────────────────────────────
def load_data():
    """
    Loads raw data from CSV.
    Returns a pandas DataFrame.
    """
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ── Clean data ──────────────────────────────────────────
def clean_data(df):
    """
    Cleans the raw DataFrame.

    Steps based on EDA findings:
    1. Drop useless row number column
    2. Create missing value indicator columns
    3. Fill missing values with 'unknown'
    4. Encode target variable as numbers
    """

    print("\nCleaning data...")

    # ── Step 1: Drop useless column ────────────────────
    # From EDA: Unnamed: 0 is just a row number
    # It has zero predictive value
    df = df.drop(columns=['Unnamed: 0'])
    print("  Dropped Unnamed: 0")

    # ── Step 2: Create indicator columns ───────────────
    # From EDA: missingness in these columns correlates
    # with better credit risk — it carries information
    # We capture that signal with binary indicator columns
    # 1 means the value was missing, 0 means it was present
    df['saving_missing'] = df['Saving accounts'].isnull().astype(int)
    df['checking_missing'] = df['Checking account'].isnull().astype(int)
    print(f"  Created saving_missing: {df['saving_missing'].sum()} missing rows flagged")
    print(f"  Created checking_missing: {df['checking_missing'].sum()} missing rows flagged")

    # ── Step 3: Fill missing values ─────────────────────
    # From EDA: missing values are informative so we
    # fill with 'unknown' — a separate category that
    # the model can learn from
    # We already captured the missingness signal above
    df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
    df['Checking account'] = df['Checking account'].fillna('unknown')
    print("  Filled missing values with 'unknown'")

    # ── Step 4: Encode target variable ─────────────────
    # Models need numbers not text
    # good = 0 (no default), bad = 1 (default)
    # We use 1 for bad because 1 conventionally means
    # the positive class — the thing we're trying to detect
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'good': 0, 'bad': 1})
    print(f"  Encoded target: good=0, bad=1")
    print(f"  Class distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")

    return df
# ── Clean data ──────────────────────────────────────────
# ── Prepare features ────────────────────────────────────
def prepare_features(df):
    """
    Separates features (X) from target (y).

    X contains everything the model uses to predict.
    y contains what we're trying to predict.

    Returns X and y separately.
    """

    # All columns except the target
    feature_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [
        'saving_missing',
        'checking_missing'
    ]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {feature_columns}")

    return X, y

# ── Main preprocessing function ─────────────────────────
def preprocess():
    """
    Full preprocessing pipeline.

    Loads raw data → cleans it → prepares features.
    Returns X and y ready for training.
    """

    print("=" * 50)
    print("Starting preprocessing pipeline")
    print("=" * 50)

    # Load
    df = load_data()

    # Clean
    df = clean_data(df)

    # Prepare features
    X, y = prepare_features(df)

    print("\n" + "=" * 50)
    print("Preprocessing complete")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Default rate: {y.mean():.1%}")
    print("=" * 50)

    return X, y


# ── Entry point ─────────────────────────────────────────
if __name__ == '__main__':
    X, y = preprocess()