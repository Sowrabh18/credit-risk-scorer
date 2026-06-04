# ─────────────────────────────────────────────────────
# tests/test_predict.py
#
# Automated tests for the credit risk prediction function.
# These run automatically on every git push via CI/CD.
# If any test fails — deployment stops. Broken code
# never reaches the AWS server.
# ─────────────────────────────────────────────────────

from src.predict import predict


# ── Sample applications for testing ────────────────────

# High risk applicant — young, large loan, long duration
HIGH_RISK = {
    'Age': 22,
    'Sex': 'male',
    'Job': 0,
    'Housing': 'rent',
    'Saving accounts': 'little',
    'Checking account': 'little',
    'Credit amount': 15000,
    'Duration': 60,
    'Purpose': 'car'
}

# Low risk applicant — older, small loan, short duration,
# missing account info (which we found in EDA is a good signal)
LOW_RISK = {
    'Age': 55,
    'Sex': 'female',
    'Job': 3,
    'Housing': 'own',
    'Saving accounts': None,
    'Checking account': None,
    'Credit amount': 1000,
    'Duration': 6,
    'Purpose': 'repairs'
}


# ── Tests ────────────────────────────────────────────────

def test_high_risk_prediction():
    """
    Young applicant with large loan and little savings
    should be predicted as bad risk.
    """
    result = predict(HIGH_RISK)
    assert result['prediction'] == 'bad', \
        f"Expected bad risk but got {result['prediction']}"


def test_low_risk_prediction():
    """
    Older applicant with small loan and missing account info
    should be predicted as good risk.
    EDA showed missing account info = better credit risk.
    """
    result = predict(LOW_RISK)
    assert result['prediction'] == 'good', \
        f"Expected good risk but got {result['prediction']}"


def test_result_has_required_fields():
    """
    Every prediction must return all four required fields.
    If any field is missing the API will crash.
    """
    result = predict(HIGH_RISK)
    assert 'prediction' in result
    assert 'probability' in result
    assert 'risk_score' in result
    assert 'top_factors' in result


def test_prediction_is_valid_label():
    """
    Prediction must always be either good or bad.
    Never anything else.
    """
    result = predict(HIGH_RISK)
    assert result['prediction'] in ['good', 'bad']


def test_probability_is_between_0_and_1():
    """
    Probability must always be a valid number between 0 and 1.
    Anything outside this range means something broke.
    """
    result = predict(HIGH_RISK)
    assert 0 <= result['probability'] <= 1


def test_risk_score_matches_probability():
    """
    Risk score is just probability as a percentage.
    They must always match each other.
    """
    result = predict(HIGH_RISK)
    assert abs(result['risk_score'] - result['probability'] * 100) < 0.1


def test_top_factors_has_five_items():
    """
    We always return exactly 5 top factors.
    Not 4, not 6 — always 5.
    """
    result = predict(HIGH_RISK)
    assert len(result['top_factors']) == 5


def test_each_factor_has_required_fields():
    """
    Each factor in top_factors must have three fields:
    feature name, impact score, and direction.
    """
    result = predict(HIGH_RISK)
    for factor in result['top_factors']:
        assert 'feature' in factor
        assert 'impact' in factor
        assert 'direction' in factor


def test_direction_is_valid():
    """
    Direction must always be one of two values.
    Either increases risk or decreases risk.
    Never anything else.
    """
    result = predict(HIGH_RISK)
    for factor in result['top_factors']:
        assert factor['direction'] in [
            'increases risk',
            'decreases risk'
        ]


def test_missing_account_handled():
    """
    Missing Saving accounts and Checking account
    should not crash the prediction function.
    This tests our missing value handling works correctly.
    """
    result = predict(LOW_RISK)
    assert result['prediction'] in ['good', 'bad']
    assert result['probability'] is not None