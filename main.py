import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import google.generativeai as genai
from dotenv import load_dotenv
from functools import lru_cache
import traceback

# =========================
# 1) Environment & Gemini
# =========================
load_dotenv()

try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ Gemini configured successfully.")
except Exception as e:
    print(f"⚠️ Could not configure Gemini: {e}")
    gemini_model = None

# =========================
# 2) FastAPI + CORS
# =========================
app = FastAPI(title="Baymax Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 3) ML Models (lazy-loaded)
# =========================
@lru_cache(maxsize=1)
def get_models():
    """
    Load ML models once on first call, cache them afterwards.
    """
    print("📦 Loading ML models (first call only)...")
    preprocessor = joblib.load("preprocessor.pkl")
    stacked_model = joblib.load("stacked_model.pkl")
    autoencoder = load_model("autoencoder.h5", compile=False)
    print("✅ Models loaded.")
    return preprocessor, stacked_model, autoencoder

# =========================
# 4) Request Schemas
# =========================
class PredictionRequest(BaseModel):
    provider_id: str

class ExplanationRequest(BaseModel):
    provider_id: str
    risk_score: int
    flagged_claims: int
    total_claims: int
    risk_level: str
    user_question: str

# =========================
# 5) Root Route
# =========================
@app.get("/")
def read_root():
    return {"status": "Baymax Fraud Detection API is running"}

# =========================
# 6) Predict Route (no SQL for now)
# =========================
@app.post("/predict")
async def predict_fraud(request: PredictionRequest):
    provider_id = request.provider_id

    # Dummy provider data (no SQL dependency)
    provider_data = pd.DataFrame([{
        "Provider": provider_id,
        "UniqueDiagnoses": 5,
        "UniqueProcedures": 3,
        "UniquePhysicians": 2,
        "TotalClaims": 100
    }])

    preprocessor, stacked_model, autoencoder = get_models()

    # Simple features for demo purposes
    features_df = provider_data.drop(columns=["Provider"]).iloc[[0]]

    # Transform features
    X_prep = preprocessor.transform(features_df)

    # Autoencoder reconstruction error
    recon = autoencoder.predict(X_prep)
    recon_err = np.mean(np.square(X_prep - recon), axis=1)

    # Final features = original + reconstruction error
    X_final = np.hstack([X_prep, recon_err.reshape(-1, 1)])

    prediction_proba = stacked_model.predict_proba(X_final)[:, 1]
    risk_score = int(prediction_proba[0] * 100)

    if risk_score >= 70:
        risk_level = "High Risk"
    elif risk_score >= 40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    total_claims = int(provider_data["TotalClaims"].iloc[0])
    flagged_claims = int(total_claims * (risk_score / 100))

    return {
        "providerId": provider_id,
        "riskScore": risk_score,
        "riskLevel": risk_level,
        "totalClaims": total_claims,
        "flaggedClaims": flagged_claims,
    }

# =========================
# 7) Explain Route
# =========================
@app.post("/explain")
async def explain_fraud(request: ExplanationRequest):
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini API not configured.")

    prompt = f"""
    You are BAYMAX, a fraud analyst.
    **Provider ID:** {request.provider_id}
    **Risk Score:** {request.risk_score}%
    **Risk Level:** {request.risk_level}
    **Total Claims:** {request.total_claims}
    **Flagged Claims:** {request.flagged_claims}

    User question: "{request.user_question}"

    Provide a clear, simple explanation of why this provider may be fraudulent.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return {"explanation": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")
