import os
import joblib
import pandas as pd
import numpy as np
import pyodbc
from fastapi import Request, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import google.generativeai as genai
from dotenv import load_dotenv
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
    print("Gemini configured successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not configure Gemini. Details: {e}")
    gemini_model = None

# =========================
# 2) FastAPI + CORS
# =========================
app = FastAPI(title="Baymax Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 3) Azure SQL (pyodbc)
# =========================
# Read from .env if present, otherwise fallback to your provided values.
DB_SERVER = os.getenv("DB_SERVER", "frauddetection.database.windows.net")
DB_NAME = os.getenv("DB_NAME", "HospitalClaimsDB")
DB_USER = os.getenv("DB_USER", "sqladmin")  # <-- Azure SQL user, NOT 'user@server'
DB_PASSWORD = os.getenv("DB_PASSWORD", "Pinkranger@123")

# IMPORTANT: Azure SQL typical UID is just the SQL login (e.g., 'sqladmin').
# Do NOT append '@servername' to UID in pyodbc; the server is provided in SERVER.
ODBC_CONN_STR = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER=tcp:{DB_SERVER},1433;"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};"
    f"PWD={DB_PASSWORD};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)

def get_conn():
    """Open a fresh pyodbc connection (recommended per-request for FastAPI)."""
    return pyodbc.connect(ODBC_CONN_STR)
def get_provider_data(provider_id: str):
    """
    Fetch provider-specific claims directly from Azure SQL.
    """
    query = "SELECT * FROM dbo.Claims WHERE Provider = ?"
    with get_conn() as conn:
        return pd.read_sql(query, conn, params=[provider_id])
# Optional: test connection on startup (won't crash app if it fails)
try:
    with get_conn() as _c:
        with _c.cursor() as _cur:
            _cur.execute("SELECT 1")
            print("✅ Azure SQL connectivity OK.")
except Exception as e:
    print("⚠️ Azure SQL connectivity test failed:", e)

# =========================
# 4) ML Models & Dataset
# =========================
try:
    preprocessor = joblib.load("preprocessor.pkl")
    stacked_model = joblib.load("stacked_model.pkl")
    autoencoder = load_model("autoencoder.h5", compile=False)
   df_full = None   # no longer loading dataset locally
    print("ML models and dataset loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Could not load a required file: {e}. Make sure all .pkl, .h5, and .csv files are present.")
    preprocessor = stacked_model = autoencoder = df_full = None

# =========================
# 5) Schemas
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
# 6) Feature Engineering
# =========================
def create_provider_features(df):
    diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
    proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
    physician_cols = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]

    def get_diversity(df_group, cols):
        codes = df_group[cols].stack().dropna().astype(str).str.lower()
        codes = codes[~codes.isin(["unknown", "0", "0.0"])]
        return codes.nunique()

    diag_div = df.groupby("Provider").apply(lambda x: get_diversity(x, diag_cols), include_groups=False).reset_index(name="UniqueDiagnoses")
    proc_div = df.groupby("Provider").apply(lambda x: get_diversity(x, proc_cols), include_groups=False).reset_index(name="UniqueProcedures")
    phys_div = df.groupby("Provider").apply(lambda x: get_diversity(x, physician_cols), include_groups=False).reset_index(name="UniquePhysicians")
    provider_counts = df.groupby("Provider").size().reset_index(name="TotalClaims")

    merged_df = df.merge(provider_counts, on="Provider", how="left")
    merged_df = merged_df.merge(diag_div, on="Provider", how="left")
    merged_df = merged_df.merge(proc_div, on="Provider", how="left")
    merged_df = merged_df.merge(phys_div, on="Provider", how="left")

    merged_df["DiagnosisDiversityRatio"] = merged_df["UniqueDiagnoses"] / merged_df["TotalClaims"]
    merged_df["ProcedureDiversityRatio"] = merged_df["UniqueProcedures"] / merged_df["TotalClaims"]
    merged_df["PhysicianDiversityRatio"] = merged_df["UniquePhysicians"] / merged_df["TotalClaims"]

    return merged_df

# =========================
# 7) Routes
# =========================
@app.get("/")
def read_root():
    return {"status": "Baymax Fraud Detection API is running"}

@app.post("/predict")
async def predict_fraud(request: PredictionRequest):
    if df_full is None:
        raise HTTPException(status_code=500, detail="Server is not ready. Models or data not loaded.")

   provider_id = request.provider_id
provider_data = get_provider_data(provider_id)

if provider_data.empty:
    raise HTTPException(status_code=404, detail=f"Provider ID '{provider_id}' not found in SQL database.")

    provider_data_fe = create_provider_features(provider_data)

    features_df = provider_data_fe.drop(
        columns=[
            "BeneID",
            "ClaimID",
            "Provider",
            "PotentialFraud",
            "DOD",
            "DOB",
            "ClaimStartDt",
            "ClaimEndDt",
            "AdmissionDt",
            "DischargeDt",
            "ClmAdmitDiagnosisCode",
            "DiagnosisGroupCode",
        ]
        + [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
        + [f"ClmProcedureCode_{i}" for i in range(1, 7)]
        + ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
    ).iloc[[0]]

    X_prep = preprocessor.transform(features_df)

    recon = autoencoder.predict(X_prep)
    recon_err = np.mean(np.square(X_prep - recon), axis=1)

    X_final = np.hstack([X_prep, recon_err.reshape(-1, 1)])

    prediction_proba = stacked_model.predict_proba(X_final)[:, 1]
    risk_score = int(prediction_proba[0] * 100)

    if risk_score >= 70:
        risk_level = "High Risk"
    elif risk_score >= 40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    total_claims = len(provider_data)
    flagged_claims = int(total_claims * (risk_score / 100) * (np.random.uniform(0.8, 1.2)))

    return {
        "providerId": provider_id,
        "riskScore": risk_score,
        "riskLevel": risk_level,
        "totalClaims": total_claims,
        "flaggedClaims": flagged_claims,
    }

@app.post("/explain")
async def explain_fraud(request: ExplanationRequest):
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini API is not configured on the server.")

    prompt = f"""
    You are an expert fraud analyst named BAYMAX.
    A healthcare provider has been analyzed for potential insurance fraud. Your task is to provide a concise, expert explanation.

    **Analysis Data:**
    - **Provider ID:** {request.provider_id}
    - **Calculated Risk Score:** {request.risk_score}%
    - **Risk Level:** {request.risk_level}
    - **Total Claims Submitted:** {request.total_claims}
    - **Potentially Fraudulent Claims:** {request.flagged_claims}

    **User's Question:** "{request.user_question}"

    Based on the data and the user's question, generate a detailed but easy-to-understand analysis.
    Focus on potential reasons WHY this provider is flagged, given the high risk score.
    Mention possible fraud schemes like **upcoding**, **phantom billing**, or **unbundling**.
    Structure your response clearly and use markdown bolding for emphasis.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return {"explanation": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {e}")

@app.post("/submit-claim")
async def submit_claim(request: Request):
    """
    Pure pyodbc insert into dbo.Claims
    Expects payload (matching your current frontend):
    {
      "providerId": "...",
      "insAmount": 123.45,
      "dedAmount": 50.00,
      "details": "...",
      "attendingPhysician": "...",
      "operatingPhysician": "...",
      "otherPhysician": "...",
      "gender": "Male" | "Female" | null,
      "race": "White" | "Black" | "Other" | "Hispanic" | null,
      "renal": 0 | 1,
      "conditions": [ ... ],
      "procedures": [ ... ],
      "diagnoses": [ ... ]
    }
    """
    try:
        data = await request.json()
        print("📥 Incoming Payload:", data)

        # --- map frontend → DB columns (aligned with your JS) ---
        providerId = data.get("providerId")
        insAmount = data.get("insuranceAmount", data.get("insAmount"))
        dedAmount = data.get("deductibleAmount", data.get("dedAmount"))
        details = data.get("claimDetails", data.get("details"))
        if not details:
            details = "N/A"   # fallback so SQL doesn’t reject

        gender = data.get("gender") or "Unknown"
        race   = data.get("race") or "Unknown"


        attending = data.get("attendingPhysician")
        operating = data.get("operatingPhysician")
        other     = data.get("otherPhysician")

        gender = data.get("gender")
        race   = data.get("race")

        renal = 1 if data.get("renalIndicator", data.get("renal")) else 0

        conditions = ",".join(data.get("conditions", [])) if data.get("conditions") else None
        procedures = ",".join(data.get("procCodes", data.get("procedures", []))) if data.get("procCodes") or data.get("procedures") else None
        diagnoses  = ",".join(data.get("diagCodes", data.get("diagnoses", []))) if data.get("diagCodes") or data.get("diagnoses") else None




        # Basic server-side validation
        if not providerId or insAmount is None or dedAmount is None or not details:
            raise HTTPException(status_code=400, detail="providerId, insAmount, dedAmount, and details are required.")

        insert_sql = """
            INSERT INTO dbo.Claims
                (ProviderId, InsAmount, DedAmount, Details,
                 AttendingPhysician, OperatingPhysician, OtherPhysician,
                 Gender, Race, Renal, Conditions, Procedures, Diagnoses)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = [
            providerId,
            float(insAmount),
            float(dedAmount),
            details,
            attending,
            operating,
            other,
            gender,
            race,
            renal,
            conditions,
            procedures,
            diagnoses,
        ]

        with get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(insert_sql, params)
                conn.commit()

        return {"status": "success", "message": "Claim saved successfully"}

    except HTTPException:
        # re-raise FastAPI HTTP exceptions
        raise
    except Exception as e:
        print("🔥 SQL ERROR:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

