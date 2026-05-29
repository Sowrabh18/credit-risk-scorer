# ─────────────────────────────────────────────────────
# main.py
#
# FastAPI application for credit risk scoring.
# Exposes the model as a REST API.
#
# Endpoints:
# GET  /        → health check
# POST /predict → credit risk prediction with explanation
#
# Run with: uvicorn main:app --reload
# ─────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from src.predict import predict

# ── Create FastAPI app ──────────────────────────────────
app = FastAPI(
    title="Credit Risk Scorer API",
    description="Predicts credit risk for loan applications with SHAP explanation",
    version="1.0.0"
)

# ── Request model ───────────────────────────────────────
# Defines exactly what a loan application looks like
# FastAPI validates every incoming request against this
# If a required field is missing — FastAPI returns an error automatically
class LoanApplication(BaseModel):
    Age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Applicant age in years"
    )
    Sex: str = Field(
        ...,
        description="Applicant sex: male or female"
    )
    Job: int = Field(
        ...,
        ge=0,
        le=3,
        description="Job type: 0=unskilled non-resident, 1=unskilled resident, 2=skilled, 3=highly skilled"
    )
    Housing: str = Field(
        ...,
        description="Housing situation: own, free, or rent"
    )
    Saving_accounts: Optional[str] = Field(
        None,
        description="Savings level: little, moderate, quite rich, rich. Leave empty if unknown."
    )
    Checking_account: Optional[str] = Field(
        None,
        description="Checking account level: little, moderate, rich. Leave empty if unknown."
    )
    Credit_amount: int = Field(
        ...,
        gt=0,
        description="Loan amount in Deutsche Marks"
    )
    Duration: int = Field(
        ...,
        gt=0,
        description="Loan duration in months"
    )
    Purpose: str = Field(
        ...,
        description="Loan purpose: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 25,
                "Sex": "male",
                "Job": 2,
                "Housing": "own",
                "Saving_accounts": "little",
                "Checking_account": "moderate",
                "Credit_amount": 5000,
                "Duration": 24,
                "Purpose": "car"
            }
        }

# ── Response models ─────────────────────────────────────
# Defines exactly what we send back
# FastAPI validates our output against this automatically

class Factor(BaseModel):
    feature: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_score: float
    top_factors: List[Factor]

# ── Home endpoint ───────────────────────────────────────
@app.get("/")
def home():
    """
    Health check endpoint.
    Confirms the API is running.
    """
    return {
        "message": "Credit Risk Scorer API is running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "docs": "GET /docs"
        }
    }

# ── Predict endpoint ────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(application: LoanApplication):
    """
    Predicts credit risk for a loan application.

    Returns:
    - prediction: good or bad credit risk
    - probability: probability of bad risk (0 to 1)
    - risk_score: probability as percentage
    - top_factors: top 5 features driving the decision
                   with direction and impact
    """

    # Validate Sex field
    if application.Sex not in ['male', 'female']:
        raise HTTPException(
            status_code=400,
            detail="Sex must be 'male' or 'female'"
        )

    # Validate Housing field
    if application.Housing not in ['own', 'free', 'rent']:
        raise HTTPException(
            status_code=400,
            detail="Housing must be 'own', 'free', or 'rent'"
        )

    # Validate Purpose field
    valid_purposes = [
        'car', 'furniture/equipment', 'radio/TV',
        'domestic appliances', 'repairs', 'education',
        'business', 'vacation/others'
    ]
    if application.Purpose not in valid_purposes:
        raise HTTPException(
            status_code=400,
            detail=f"Purpose must be one of: {valid_purposes}"
        )

    # Convert request to dictionary
    # Map underscore field names back to space names
    # that match what the model was trained on
    application_dict = {
        'Age': application.Age,
        'Sex': application.Sex,
        'Job': application.Job,
        'Housing': application.Housing,
        'Saving accounts': application.Saving_accounts,
        'Checking account': application.Checking_account,
        'Credit amount': application.Credit_amount,
        'Duration': application.Duration,
        'Purpose': application.Purpose
    }

    # Get prediction from predict function
    result = predict(application_dict)

    return result