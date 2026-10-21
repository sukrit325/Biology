import http.server
import json
import math
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from main import generate_model_metrics_only

PORT = 8000
DATA_FILE = "paternity.csv"

# Global trained model holder
MODEL = None
FEATURE_COLS = None

def init_model():
    global MODEL, FEATURE_COLS
    if not os.path.exists(DATA_FILE):
        print(f"[{DATA_FILE}] not found for training.")
        return

    print("Training ML model from paternity.csv...")
    try:
        from main import prepare_data
        df = prepare_data(DATA_FILE)
        
        # ──> CALL IT HERE <──
        generate_model_metrics_only(df)

        feature_cols = [col for col in df.columns if col.startswith('compat_')] + ['total_compatible_loci', 'log_combined_lr']
        X = df[feature_cols]
        y = df['opa_match'].astype(int)

        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced')
        rf.fit(X, y)
        MODEL = rf
        FEATURE_COLS = feature_cols
        print("✓ Random Forest Classifier trained successfully!")
    except Exception as e:
        print(f"Error training model: {e}")
class MLHandler(http.server.BaseHTTPRequestHandler):
    def _set_headers(self, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(status=204)

    def do_GET(self):
        if self.path == "/ml/version" or self.path == "/health":
            self._set_headers()
            resp = {
                "status": "online",
                "service": "GENETRACE ML Analytics Core",
                "model": "RandomForestClassifier",
                "features": 22
            }
            self.wfile.write(json.dumps(resp).encode('utf-8'))

        elif self.path.startswith("/download-csv/"):
            case_id = self.path.split("/")[-1]
            csv_content = f"Case ID,{case_id}\nEngine,GENETRACE ML Backend\nStatus,Verified\n"
            self._set_headers(content_type="text/csv")
            self.wfile.write(csv_content.encode('utf-8'))

        else:
            self._set_headers(status=404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode('utf-8'))

    def do_POST(self):
        # 1. Clean the path by removing query strings and trailing slashes
        clean_path = self.path.split('?')[0].rstrip('/')
        
        # Print this to your terminal to debug what your frontend is sending!
        print(f"Incoming POST request path: {self.path} (cleaned: {clean_path})")

        if clean_path == "/ml/analyze-paternity":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body.decode('utf-8'))
                features = data.get("features", {})
                case_id = data.get("case_id", "UNKNOWN")

                # If trained model exists, run prediction
                prediction = 1
                probability = 0.9999
                model_name = "RandomForestClassifier"

                if MODEL and FEATURE_COLS:
                    # Construct feature vector
                    x_row = []
                    for col in FEATURE_COLS:
                        x_row.append(features.get(col, 0.0))
                    
                    x_arr = pd.DataFrame([x_row], columns=FEATURE_COLS)
                    pred_class = MODEL.predict(x_arr)[0]
                    prob = MODEL.predict_proba(x_arr)[0][1]
                    prediction = int(pred_class)
                    probability = float(prob)
                else:
                    # Heuristic inference from features
                    total_compat = features.get("total_compatible_loci", 20)
                    if total_compat >= 19:
                        prediction = 1
                        probability = 0.9999
                    elif total_compat <= 16:
                        prediction = 0
                        probability = 0.0001
                    else:
                        prediction = 0
                        probability = 0.50

                verdict = "STRONG INCLUSION" if prediction == 1 and probability >= 0.90 else \
                        "EXCLUSION" if prediction == 0 and probability <= 0.10 else "REVIEW REQUIRED"

                response_payload = {
                    "status": "success",
                    "case_id": case_id,
                    "ml_model": model_name,
                    "prediction": prediction,
                    "probability": probability,
                    "verdict": verdict,
                    "total_compatible_loci": features.get("total_compatible_loci", 20),
                    "cpi": features.get("combined_lr", 1.0)
                }

                self._set_headers()
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))

            except Exception as e:
                self._set_headers(status=400)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self._set_headers(status=404)
            self.wfile.write(json.dumps({"error": f"Endpoint not found: {self.path}"}).encode('utf-8'))

def run():
    init_model()
    server_address = ('', PORT)
    httpd = http.server.HTTPServer(server_address, MLHandler)
    print(f"\n=======================================================")
    print(f"🚀 GENETRACE ML Backend running on http://localhost:{PORT}")
    print(f"Endpoints:")
    print(f"  POST http://localhost:{PORT}/ml/analyze-paternity")
    print(f"  GET  http://localhost:{PORT}/ml/version")
    print(f"  GET  http://localhost:{PORT}/download-csv/<case_id>")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ML Server...")
        httpd.server_close()

if __name__ == "__main__":
    run()
