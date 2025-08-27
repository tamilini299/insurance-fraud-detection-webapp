import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# ---- 1. Data Load & Split by Provider ----
def load_and_split_data(file_path):
    print("-- Loading data --")
    # CRITICAL FIX: Tell pandas to treat these columns as strings
    dtype_dict = {
        'RenalDiseaseIndicator': 'str',
        'Gender': 'str',
        'Race': 'str',
        'ChronicCond_Alzheimer': 'str',
        'ChronicCond_Heartfailure': 'str',
        'ChronicCond_KidneyDisease': 'str',
        'ChronicCond_Cancer': 'str',
        'ChronicCond_ObstrPulmonary': 'str',
        'ChronicCond_Stroke': 'str',
        'ChronicCond_Osteoporasis': 'str',
        'ChronicCond_rheumatoidarthritis': 'str',
        'ChronicCond_contdHeartfailure': 'str',
        'ChronicCond_Depression': 'str'
    }
    df = pd.read_csv(file_path, low_memory=False, dtype=dtype_dict)
    
    print(f"Columns: {list(df.columns)}")
    df['IsFraud'] = df['PotentialFraud'].apply(lambda x: 1 if x == 'Yes' else 0)
    providers = df['Provider'].unique()
    providers_train, providers_test = train_test_split(
        providers, test_size=0.2, random_state=42
    )
    train_df = df[df['Provider'].isin(providers_train)].reset_index(drop=True)
    test_df = df[df['Provider'].isin(providers_test)].reset_index(drop=True)
    print(f"Train rows: {train_df.shape[0]}, Test rows: {test_df.shape}")
    return train_df, test_df

# ---- 2. Provider-Level Feature Engineering (NO target aggregates) ----
def create_provider_features(df):
    df['IsFraud'] = df['PotentialFraud'].map({'Yes':1, 'No':0})
    
    diag_cols = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
    proc_cols = [f'ClmProcedureCode_{i}' for i in range(1, 7)]
    physician_cols = ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']

    def get_diversity(df_group, cols):
        codes = df_group[cols].stack().dropna().astype(str).str.lower()
        codes = codes[~codes.isin(['unknown', '0', '0.0'])]
        return codes.nunique()

    diag_div = df.groupby('Provider').apply(lambda x: get_diversity(x, diag_cols), include_groups=False).reset_index(name='UniqueDiagnoses')
    proc_div = df.groupby('Provider').apply(lambda x: get_diversity(x, proc_cols), include_groups=False).reset_index(name='UniqueProcedures')
    phys_div = df.groupby('Provider').apply(lambda x: get_diversity(x, physician_cols), include_groups=False).reset_index(name='UniquePhysicians')

    provider_counts = df.groupby('Provider').size().reset_index(name='TotalClaims')

    df = df.merge(provider_counts, on='Provider', how='left')
    df = df.merge(diag_div, on='Provider', how='left')
    df = df.merge(proc_div, on='Provider', how='left')
    df = df.merge(phys_div, on='Provider', how='left')

    df['DiagnosisDiversityRatio'] = df['UniqueDiagnoses'] / df['TotalClaims']
    df['ProcedureDiversityRatio'] = df['UniqueProcedures'] / df['TotalClaims']
    df['PhysicianDiversityRatio'] = df['UniquePhysicians'] / df['TotalClaims']
    return df

# ---- 3. Prepare ML data ----
def get_ml_ready(df):
    drop_cols = [
        'BeneID', 'ClaimID', 'Provider', 'PotentialFraud', 'DOD', 'DOB', 'ClaimStartDt',
        'ClaimEndDt', 'AdmissionDt', 'DischargeDt', 'ClmAdmitDiagnosisCode', 'DiagnosisGroupCode'
    ] + [f'ClmDiagnosisCode_{i}' for i in range(1,11)] + [f'ClmProcedureCode_{i}' for i in range(1,7)] \
      + ['AttendingPhysician','OperatingPhysician','OtherPhysician',
          'FraudulentClaims', 'FraudRate'
      ]
    features = [col for col in df.columns if col not in (drop_cols + ['IsFraud'])]
    X = df[features]
    y = df['IsFraud']
    return X, y

# ---- 4. Model pipeline including autoencoder, XGBoost & CatBoost + Stacking ----
def run_hybrid_model(X_train, y_train, X_test, y_test):
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ], remainder='passthrough')

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    print("-- Training autoencoder on non-fraud claims --")
    X_train_nonfraud = X_train_prep[y_train == 0]
    input_dim = X_train_nonfraud.shape[1]  #number of features in data
    encoding_dim = max(1, input_dim // 2)
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation='relu')(input_layer) #compresses the input into fewer numbers
    decoder = Dense(input_dim, activation='sigmoid')(encoder) #tries to reconstruct the original input from the compressed numbers.
    autoencoder = Model(input_layer, decoder)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    autoencoder.fit(X_train_nonfraud, X_train_nonfraud, epochs=10, batch_size=64, verbose=0)
    
    print("-- Saving autoencoder model --")
    autoencoder.save('autoencoder.h5')

    train_recon = autoencoder.predict(X_train_prep)
    test_recon = autoencoder.predict(X_test_prep)

    train_recon_err = np.mean(np.square(X_train_prep - train_recon), axis=1)
    test_recon_err = np.mean(np.square(X_test_prep - test_recon), axis=1)

    X_train_final = np.hstack([X_train_prep, train_recon_err.reshape(-1, 1)])
    X_test_final = np.hstack([X_test_prep, test_recon_err.reshape(-1, 1)])

    xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    cat = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, eval_metric='F1', random_seed=42, verbose=0)
    meta_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    stack = StackingClassifier(estimators=[('xgb', xgb), ('cat', cat)], final_estimator=meta_model, cv=5, stack_method="predict_proba", n_jobs=-1)

    print("-- Training Stacking Ensemble (XGB + CatBoost → Logistic Regression) --")
    stack.fit(X_train_final, y_train)
    y_pred_stack = stack.predict(X_test_final)

    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(stack, 'stacked_model.pkl')
    print("Saved preprocessor and Stacked model.")

    print("\n--- Stacked Ensemble Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_stack):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_stack):.4f}")
    print(classification_report(y_test, y_pred_stack))

    cm = confusion_matrix(y_test, y_pred_stack)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title("Stacked Ensemble Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("stacked_confusion_matrix.png")
    print("Saved stacked_confusion_matrix.png")

if __name__ == "__main__":
    try:
        data_file = 'Final_Fraud_Dataset_Fixed.csv'
        train_df, test_df = load_and_split_data(data_file)
        
        train_fe = create_provider_features(train_df)
        test_fe = create_provider_features(test_df)

        X_train, y_train = get_ml_ready(train_fe)
        X_test, y_test = get_ml_ready(test_fe)

        run_hybrid_model(X_train, y_train, X_test, y_test)

    except FileNotFoundError:
        print(f"File {data_file} not found. Please check file path.")