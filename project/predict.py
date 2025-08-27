import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from pathlib import Path
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- Helper Functions (Recreated from catxg.py) ---
def create_provider_features(df):
    """Recreates the provider-level features from the original training script."""
    diag_cols = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
    proc_cols = [f'ClmProcedureCode_{i}' for i in range(1, 7)]
    physician_cols = ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']

    def get_diversity(df_group, cols):
        codes = df_group[cols].stack().dropna().astype(str).str.lower()
        codes = codes[~codes.isin(['unknown', '0', '0.0'])]
        return codes.nunique()

    diag_div = df.groupby('Provider').apply(
        lambda x: get_diversity(x, diag_cols), include_groups=False
    ).reset_index(name='UniqueDiagnoses')
    
    proc_div = df.groupby('Provider').apply(
        lambda x: get_diversity(x, proc_cols), include_groups=False
    ).reset_index(name='UniqueProcedures')
    
    phys_div = df.groupby('Provider').apply(
        lambda x: get_diversity(x, physician_cols), include_groups=False
    ).reset_index(name='UniquePhysicians')
    
    provider_counts = df.groupby('Provider').size().reset_index(name='TotalClaims')

    df = df.merge(provider_counts, on='Provider', how='left')
    df = df.merge(diag_div, on='Provider', how='left')
    df = df.merge(proc_div, on='Provider', how='left')
    df = df.merge(phys_div, on='Provider', how='left')

    df['DiagnosisDiversityRatio'] = df['UniqueDiagnoses'] / df['TotalClaims']
    df['ProcedureDiversityRatio'] = df['UniqueProcedures'] / df['TotalClaims']
    df['PhysicianDiversityRatio'] = df['UniquePhysicians'] / df['TotalClaims']
    return df

def get_ml_ready(df):
    """Extracts the features needed for the ML model."""
    drop_cols = [
        'BeneID', 'ClaimID', 'Provider', 'PotentialFraud', 'DOD', 'DOB', 'ClaimStartDt',
        'ClaimEndDt', 'AdmissionDt', 'DischargeDt', 'ClmAdmitDiagnosisCode', 'DiagnosisGroupCode'
    ] + [f'ClmDiagnosisCode_{i}' for i in range(1, 11)] + [f'ClmProcedureCode_{i}' for i in range(1, 7)] \
      + ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician', 'FraudulentClaims', 'FraudRate', 'IsFraud']
    
    features = [col for col in df.columns if col not in drop_cols]
    X = df[features]
    return X

def get_new_claim_input(full_df_columns):
    """
    Prompts the user for a new claim's details, and fills in missing columns.
    """
    print("\n--- Enter New Claim Details ---")
    
    # List of columns the user will manually input
    user_input_features = {
        'InscClaimAmtReimbursed': 'float', 'DeductibleAmtPaid': 'float', 'Gender': 'str',
        'Race': 'str', 'RenalDiseaseIndicator': 'str', 'ChronicCond_Alzheimer': 'str',
        'ChronicCond_Heartfailure': 'str', 'ChronicCond_KidneyDisease': 'str',
        'ChronicCond_Cancer': 'str', 'ChronicCond_ObstrPulmonary': 'str',
        'ChronicCond_Stroke': 'str', 'ChronicCond_Osteoporasis': 'str',
        'ChronicCond_rheumatoidarthritis': 'str', 'ChronicCond_contdHeartfailure': 'str',
        'ChronicCond_Depression': 'str', 'ClmProcedureCode_1': 'str',
        'ClmProcedureCode_2': 'str', 'ClmProcedureCode_3': 'str',
        'ClmProcedureCode_4': 'str', 'ClmProcedureCode_5': 'str',
        'ClmProcedureCode_6': 'str', 'ClmDiagnosisCode_1': 'str',
        'ClmDiagnosisCode_2': 'str', 'ClmDiagnosisCode_3': 'str',
        'ClmDiagnosisCode_4': 'str', 'ClmDiagnosisCode_5': 'str',
        'ClmDiagnosisCode_6': 'str', 'ClmDiagnosisCode_7': 'str',
        'ClmDiagnosisCode_8': 'str', 'ClmDiagnosisCode_9': 'str',
        'ClmDiagnosisCode_10': 'str'
    }

    user_data = {}
    for feature, data_type in user_input_features.items():
        while True:
            try:
                user_input = input(f"Enter {feature} ({data_type}): ")
                if data_type == 'float':
                    user_data[feature] = float(user_input)
                elif data_type == 'int':
                    user_data[feature] = int(user_input)
                else:
                    user_data[feature] = user_input
                break
            except ValueError:
                print(f"Invalid input. Please enter a valid {data_type}.")

    user_df = pd.DataFrame([user_data])
    
    for col in full_df_columns:
        if col not in user_df.columns:
            user_df[col] = np.nan
    
    return user_df.iloc[0]

def predict_fraud(new_claim_data, provider_id, full_df, preprocessor, stacked_model, autoencoder):
    """
    Predicts fraud for a single new claim by integrating it with historical provider data.
    """
    try:
        new_claim_df = pd.DataFrame([new_claim_data]).astype(
            dtype={'RenalDiseaseIndicator': 'object', 'Gender': 'object', 'Race': 'object'}
        )
        new_claim_df['Provider'] = provider_id
        
        historical_df = full_df[full_df['Provider'] == provider_id]
        
        combined_df = pd.concat([historical_df, new_claim_df], ignore_index=True)
        
        combined_df_fe = create_provider_features(combined_df)
        claim_for_prediction = combined_df_fe.tail(1)
        
        X_claim = get_ml_ready(claim_for_prediction)
        X_prep = preprocessor.transform(X_claim)
        
        recon = autoencoder.predict(X_prep, verbose=0)
        recon_err = np.mean(np.square(X_prep - recon), axis=1)
        
        X_final = np.hstack([X_prep, recon_err.reshape(-1, 1)])
        
        prediction_label = stacked_model.predict(X_final)[0]
        prediction_score = stacked_model.predict_proba(X_final)[:, 1][0]
        
        return prediction_label, prediction_score
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None, None

if __name__ == "__main__":
    try:
        print("Loading models and full dataset...")
        preprocessor = joblib.load('preprocessor.pkl')
        stacked_model = joblib.load('stacked_model.pkl')
        autoencoder = load_model('autoencoder.h5', custom_objects={'mse': MeanSquaredError()})
        full_df = pd.read_csv('Final_Fraud_Dataset_Fixed.csv', low_memory=False)
        print("Models and dataset loaded successfully.")
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")
        exit()

    provider_id_input = input("\nEnter the Provider ID for the new claim: ")
    new_claim_data = get_new_claim_input(full_df.columns)
    
    predicted_label, prediction_score = predict_fraud(
        new_claim_data, provider_id_input, full_df, preprocessor, stacked_model, autoencoder
    )
    
    if predicted_label is not None:
        prediction_result = "YES" if predicted_label == 1 else "NO"
        print("\n--- Prediction Results ---")
        print(f"Prediction for new claim with Provider ID '{provider_id_input}':")
        print(f"Fraudulent: {prediction_result}")
        if prediction_score is not None:
            print(f"Prediction Score (Confidence): {prediction_score:.4f}")