import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
from contextlib import asynccontextmanager

# ONNX Imports
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Global holders
onnx_session = None
feature_columns = []

ONNX_FILE = "paternity_model.onnx"
FEATURES_FILE = "feature_cols.pkl"
CSV_FILE = "paternity.csv"

def generate_model_metrics_only(df, output_path="model_metrics.json"):
    """Computes and saves performance metrics JSON for the frontend dashboard."""
    try:
        feature_cols = [col for col in df.columns if col.startswith('compat_')] + ['total_compatible_loci', 'log_combined_lr']
        X = df[feature_cols]
        y = df['opa_match'].astype(int)
        
        n_splits = min(5, int(y.value_counts().min()))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=89)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=98, class_weight='balanced')
        dt = DecisionTreeClassifier(max_depth=15, random_state=42)

        rf_probs = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')[:, 1]
        dt_probs = cross_val_predict(dt, X, y, cv=cv, method='predict_proba')[:, 1]
        print("Model expects:", getattr(rf_probs, "n_features_in_", "unknown"))
        print("Feature columns:", feature_columns)
        print("Training shape:", X.shape)
        rf_preds = (rf_probs >= 0.5).astype(int)
        dt_preds = (dt_probs >= 0.5).astype(int)

        def get_weighted_metrics(y_true, y_pred):
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            return {"precision": float(prec), "recall": float(rec), "f1-score": float(f1)}

        metrics = {
            "status": "success",
            "random_forest": {
                "roc_auc": float(roc_auc_score(y, rf_probs)),
                "accuracy": float(accuracy_score(y, rf_preds)),
                "weighted_avg": get_weighted_metrics(y, rf_preds)
            },
            "decision_tree": {
                "roc_auc": float(roc_auc_score(y, dt_probs)),
                "accuracy": float(accuracy_score(y, dt_preds)),
                "weighted_avg": get_weighted_metrics(y, dt_preds)
            }
        }

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        print(f"✓ '{output_path}' generated successfully!")
    except Exception as e:
        print(f"⚠️ Could not generate model_metrics.json: {e}")

# --- 1. Core Logic Functions ---

def check_locus_compatibility(m_str, f_str, c_str):
    try:
        m = set(str(m_str).split('/'))
        f = set(str(f_str).split('/'))
        c = set(str(c_str).split('/'))
    except Exception:
        return 0

    found = False
    for m_allele in m:
        rem_c = c.copy()
        if m_allele in rem_c:
            rem_c.remove(m_allele)
            if len(rem_c) == 0:
                if len(c.intersection(f)) > 0:
                    found = True
            else:
                f_allele = list(rem_c)[0]
                if f_allele in f:
                    found = True
    return 1 if found else 0


def prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = [col.strip().lower() for col in df.columns]

    if 'paternity_label' not in df.columns:
        raise ValueError("CSV must contain a 'paternity_label' column.")

    mother_cols = [col for col in df.columns if col.startswith('mother_')]
    loci = [col.replace('mother_', '') for col in mother_cols]

    for locus in loci:
        col_name = f'compat_{locus}'
        df[col_name] = [
            check_locus_compatibility(
                row.get(f'mother_{locus}'),
                row.get(f'father_{locus}'),
                row.get(f'child_{locus}')
            )
            for _, row in df.iterrows()
        ]

    compat_cols = [f'compat_{locus}' for locus in loci]
    df['total_compatible_loci'] = df[compat_cols].sum(axis=1)

    if 'combined_lr' in df.columns:
        df['log_combined_lr'] = np.log10(df['combined_lr'].clip(lower=np.finfo(float).tiny))
    else:
        df['log_combined_lr'] = 0.0

    df['opa_match'] = df['paternity_label'].astype(bool)
    return df


# --- 2. Startup Lifespan (Loads ONNX or trains/converts if missing) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global onnx_session, feature_columns
    print("Initializing ONNX model session on startup...")
    
#     # 1. Try loading pre-built ONNX model & features
    if os.path.exists(ONNX_FILE) and os.path.exists(FEATURES_FILE):
        print("Loading pre-trained ONNX model from disk...")
        try:
            onnx_session = ort.InferenceSession(ONNX_FILE)
            feature_columns = joblib.load(FEATURES_FILE)
            print("✓ ONNX model session loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load ONNX model: {e}")

    # 2. If missing, train from CSV and convert to ONNX
    if onnx_session is None and os.path.exists(CSV_FILE):
        print("Training model from paternity.csv and converting to ONNX...")
        try:
            df = prepare_data(CSV_FILE)
            feature_columns = [col for col in df.columns if col.startswith('compat_')] + ['total_compatible_loci', 'log_combined_lr']
            X = df[feature_columns]
            y = df['opa_match'].astype(int)

            # Train model
            rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=98, class_weight='balanced')
            rf.fit(X, y)

            # Convert to ONNX
            num_features = len(feature_columns)
            initial_type = [('float_input', FloatTensorType([None, num_features]))]
            options = {id(rf): {'zipmap': False}}
            onnx_model = convert_sklearn(rf, initial_types=initial_type, options=options)

            # Save ONNX and feature list
            with open(ONNX_FILE, "wb") as f:
                f.write(onnx_model.SerializeToString())
            joblib.dump(feature_columns, FEATURES_FILE)
            
            onnx_session = ort.InferenceSession(ONNX_FILE)
            print("✓ Model successfully trained, converted to ONNX, and loaded!")

            # Generate dashboard metrics JSON
            generate_model_metrics_only(df)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Error during ONNX training/conversion: {e}")
    elif onnx_session is None:
        print("⚠️ Warning: Neither ONNX files nor paternity.csv were found.")
        
    yield
    print("Application shutting down...")

# # Initialize FastAPI application with lifespan
app = FastAPI(
    title="GENETRACE ONNX ML Backend",
    description="API for STR Mendelian inheritance checking powered by ONNX Runtime.",
    lifespan=lifespan
)


# --- 3. Request Schemas ---

class LocusData(BaseModel):
    mother: str
    father: str
    child: str

class PredictionRequest(BaseModel):
    loci_data: Dict[str, LocusData]
    combined_lr: float = 1.0


# --- 4. API Endpoints ---

@app.get("/")
def home():
    return {"status": "online", "message": "GENETRACE ONNX ML Backend is running."}

@app.get("/metrics")
def get_metrics():
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="model_metrics.json not found.")

@app.post("/predict")
def predict_paternity(payload: PredictionRequest):
    """Predicts paternity likelihood using ONNX Runtime."""
    if onnx_session is None or not feature_columns:
        raise HTTPException(status_code=500, detail="ONNX model session is not initialized.")

    row_features = {}
    total_compat = 0
    
    for locus, val in payload.loci_data.items():
        compat = check_locus_compatibility(val.mother, val.father, val.child)
        row_features[f'compat_{locus}'] = compat
        total_compat += compat

    row_features['total_compatible_loci'] = total_compat
    row_features['log_combined_lr'] = np.log10(max(payload.combined_lr, np.finfo(float).tiny))

    input_df = pd.DataFrame([row_features])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  
    
    # ONNX requires explicit float32 arrays
    X_pred = input_df[feature_columns].astype(np.float32).to_numpy()

    # Run inference through ONNX Runtime
    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: X_pred})
    
    prediction = int(np.asarray(outputs[0]).reshape(-1)[0])
    probability_output = outputs[1]
    values_method = getattr(probability_output, "values", None)
    indices_method = getattr(probability_output, "indices", None)
    if callable(values_method) and callable(indices_method):
        # ONNX Runtime may return a SparseTensor, which does not support indexing.
        values = np.asarray(values_method()).reshape(-1)
        indices = np.asarray(indices_method())
        probability = 0.0
        for index, value in zip(indices.reshape(-1, indices.shape[-1]), values):
            if index[-1] == 1:
                probability = float(value)
                break
    else:
        probability_array = np.asarray(probability_output)
        probability = float(probability_array.reshape(-1, 2)[0, 1])

    return {
        "paternity_match": bool(prediction),
        "probability": probability
    }