"""
server.py
=========
Lightweight, dependency-minimal Python HTTP server for the EV Battery
Management System (BMS) Digital Twin & Interactive SOC Estimation Dashboard.

Loads:
  - Random Forest (25 trees): Random_Forest/results/models/rf_25_trees_final.joblib
  - Decision Tree:            Decision_Tree/results/models/dt_model.joblib
  - KNN (k=5 + scaler):       KNN/results/models/knn_model.joblib

Serves:
  - Static dashboard assets (index.html, style.css, app.js)
  - GET  /api/trips                     -> List 10 unseen test trips + summaries
  - GET  /api/trip/<id>?step=5          -> Downsampled telemetry with ground-truth + 3 model predictions
  - POST /api/predict                   -> Real-time inference for manual What-If sliders
  - GET  /api/benchmarks                -> Pre-computed evaluation metrics table
"""

import os
import sys
import json
import time
import glob
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np
import pandas as pd
import joblib

# -------------------------------------------------------------
# Path Resolution
# -------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "archive")

RF_MODEL_PATH = os.path.join(PROJECT_ROOT, "Random_Forest", "results", "models", "rf_25_trees_final.joblib")
DT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Decision_Tree", "results", "models", "dt_model.joblib")
KNN_MODEL_PATH = os.path.join(PROJECT_ROOT, "KNN", "results", "models", "knn_model.joblib")

# -------------------------------------------------------------
# CSV Ingestion & Header Cleaning (Self-Contained)
# -------------------------------------------------------------
CSV_SEPARATOR = ";"
CSV_ENCODING = "latin1"
PAPER_FEATURES = [
    "Battery Voltage [V]",
    "Battery Current [A]",
    "Battery Temperature [°C]",
    "Ambient Temperature [°C]"
]
TARGET_SOC = "SoC [%]"


def _clean_column_name(col: str) -> str:
    col = col.strip()
    col = col.replace("[%)", "[%]")
    col = col.replace("]]]", "]")
    return col


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [_clean_column_name(c) for c in df.columns]
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def load_single_trip(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)
    return _standardise_columns(df)


def _find_matching_column(df: pd.DataFrame, target_name: str) -> str:
    def ascii_skeleton(s: str) -> str:
        return s.encode("ascii", "ignore").decode("ascii").strip().lower()

    target_skel = ascii_skeleton(target_name)
    for col in df.columns:
        if ascii_skeleton(col) == target_skel:
            return col
    for col in df.columns:
        if target_name.split("[")[0].strip().lower() in col.lower():
            if "max" not in col.lower() and "min" not in col.lower():
                return col
    raise KeyError(f"Could not find column matching '{target_name}' in columns: {list(df.columns)}")


def extract_features_target(
    df: pd.DataFrame,
    feature_names: list = PAPER_FEATURES,
    target_name: str = TARGET_SOC
) -> pd.DataFrame:
    resolved_features = [_find_matching_column(df, f) for f in feature_names]
    resolved_target = _find_matching_column(df, target_name)
    cols_needed = resolved_features + [resolved_target]
    subset = df[cols_needed].copy()
    rename_map = {old: new for old, new in zip(resolved_features, feature_names)}
    rename_map[resolved_target] = target_name
    subset.rename(columns=rename_map, inplace=True)
    subset.dropna(inplace=True)
    for col in subset.columns:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
    subset.dropna(inplace=True)
    return subset

# -------------------------------------------------------------
# Global Model Registry
# -------------------------------------------------------------
MODELS = {
    "rf": None,
    "dt": None,
    "knn": None,
    "knn_scaler": None
}

TRIP_CACHE = {}

def load_models():
    print("=" * 60)
    print("  EV BMS Digital Twin — Loading Machine Learning Models")
    print("=" * 60)

    # 1. Random Forest
    if os.path.exists(RF_MODEL_PATH):
        t0 = time.time()
        print(f"[*] Loading Random Forest (25 Trees) from: {os.path.basename(RF_MODEL_PATH)}...")
        MODELS["rf"] = joblib.load(RF_MODEL_PATH)
        print(f"    [+] Loaded in {time.time() - t0:.2f}s")
    else:
        print(f"[!] Warning: RF model not found at {RF_MODEL_PATH}")

    # 2. Decision Tree
    if os.path.exists(DT_MODEL_PATH):
        t0 = time.time()
        print(f"[*] Loading Decision Tree from: {os.path.basename(DT_MODEL_PATH)}...")
        MODELS["dt"] = joblib.load(DT_MODEL_PATH)
        print(f"    [+] Loaded in {time.time() - t0:.2f}s")
    else:
        print(f"[!] Warning: DT model not found at {DT_MODEL_PATH}")

    # 3. KNN
    if os.path.exists(KNN_MODEL_PATH):
        t0 = time.time()
        print(f"[*] Loading KNN (k=5) from: {os.path.basename(KNN_MODEL_PATH)}...")
        bundle = joblib.load(KNN_MODEL_PATH)
        MODELS["knn"] = bundle["knn_model"]
        MODELS["knn_scaler"] = bundle["scaler"]
        print(f"    [+] Loaded in {time.time() - t0:.2f}s")
    else:
        print(f"[!] Warning: KNN model not found at {KNN_MODEL_PATH}")

    print("=" * 60)
    print("  Models successfully registered into memory.")
    print("=" * 60)


# -------------------------------------------------------------
# Trip Catalog & Downsampling
# -------------------------------------------------------------
def get_test_trips_catalog():
    """Returns summaries of the 10 unseen test trips (TripB29 - TripB38)."""
    test_files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "Trip*.csv")))[-10:]
    catalog = []
    
    trip_notes = {
        "TripB29": "Suburban commute with moderate acceleration",
        "TripB30": "Highway cruising & urban stop-and-go",
        "TripB31": "Dynamic driving with heavy regenerative braking",
        "TripB32": "High-speed expressway sprint",
        "TripB33": "Short city transit with cold ambient start",
        "TripB34": "Stop-and-go city traffic & idling intervals",
        "TripB35": "Extended mixed road profile",
        "TripB36": "Long-distance rural & highway transit",
        "TripB37": "Moderate rolling topography route",
        "TripB38": "Aggressive throttle transients & downhill braking"
    }

    for fpath in test_files:
        trip_id = os.path.splitext(os.path.basename(fpath))[0]
        try:
            df = load_single_trip(fpath)
            clean_df = extract_features_target(df)
            
            time_col = "Time [s]" if "Time [s]" in df.columns else None
            vel_col = "Velocity [km/h]" if "Velocity [km/h]" in df.columns else None
            
            duration_s = float(df[time_col].iloc[-1] - df[time_col].iloc[0]) if time_col else len(clean_df) * 0.1
            initial_soc = float(clean_df["SoC [%]"].iloc[0])
            final_soc = float(clean_df["SoC [%]"].iloc[-1])
            avg_speed = float(df[vel_col].mean()) if vel_col else 45.0

            catalog.append({
                "id": trip_id,
                "filename": os.path.basename(fpath),
                "duration_min": round(duration_s / 60.0, 1),
                "duration_s": round(duration_s, 1),
                "samples_count": len(clean_df),
                "initial_soc": round(initial_soc, 1),
                "final_soc": round(final_soc, 1),
                "soc_delta": round(initial_soc - final_soc, 1),
                "avg_speed_kmh": round(avg_speed, 1),
                "description": trip_notes.get(trip_id, "Standard drive cycle")
            })
        except Exception as e:
            print(f"[!] Error inspecting {trip_id}: {e}")

    return catalog


def load_and_process_trip(trip_id: str, step: int = 5):
    """
    Loads raw trip data, cleans columns, downsamples by `step` (default 5 = every 0.5s),
    and computes predictions for RF, DT, and KNN.
    """
    cache_key = f"{trip_id}_step_{step}"
    if cache_key in TRIP_CACHE:
        return TRIP_CACHE[cache_key]

    fpath = os.path.join(ARCHIVE_DIR, f"{trip_id}.csv")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Trip file {trip_id}.csv not found.")

    raw_df = load_single_trip(fpath)
    clean_df = extract_features_target(raw_df)

    # Downsample
    sub_clean = clean_df.iloc[::step].copy()
    sub_raw = raw_df.loc[sub_clean.index].copy()

    # Feature matrix for inference
    feature_cols = [
        "Battery Voltage [V]",
        "Battery Current [A]",
        clean_df.columns[2], # Battery Temp
        clean_df.columns[3]  # Ambient Temp
    ]
    X = sub_clean[feature_cols].values

    # RF Predictions
    pred_rf = MODELS["rf"].predict(X) if MODELS["rf"] else np.zeros(len(X))
    
    # DT Predictions
    pred_dt = MODELS["dt"].predict(X) if MODELS["dt"] else np.zeros(len(X))
    
    # KNN Predictions
    if MODELS["knn"] and MODELS["knn_scaler"]:
        X_scaled = MODELS["knn_scaler"].transform(X)
        pred_knn = MODELS["knn"].predict(X_scaled)
    else:
        pred_knn = np.zeros(len(X))

    time_col = "Time [s]" if "Time [s]" in sub_raw.columns else None
    vel_col = "Velocity [km/h]" if "Velocity [km/h]" in sub_raw.columns else None

    # Construct point sequence
    points = []
    actual_soc_arr = sub_clean["SoC [%]"].values
    voltage_arr = sub_clean[feature_cols[0]].values
    current_arr = sub_clean[feature_cols[1]].values
    batt_temp_arr = sub_clean[feature_cols[2]].values
    ambient_temp_arr = sub_clean[feature_cols[3]].values

    times = sub_raw[time_col].values if time_col else np.arange(len(X)) * (step * 0.1)
    velocities = sub_raw[vel_col].values if vel_col else np.zeros(len(X))

    for i in range(len(X)):
        points.append({
            "t": round(float(times[i]), 1),
            "v_kmh": round(float(velocities[i]), 1) if not np.isnan(velocities[i]) else 0.0,
            "volt": round(float(voltage_arr[i]), 2),
            "curr": round(float(current_arr[i]), 2),
            "b_temp": round(float(batt_temp_arr[i]), 2),
            "a_temp": round(float(ambient_temp_arr[i]), 2),
            "act_soc": round(float(actual_soc_arr[i]), 2),
            "rf_soc": round(float(pred_rf[i]), 2),
            "dt_soc": round(float(pred_dt[i]), 2),
            "knn_soc": round(float(pred_knn[i]), 2),
        })

    # Summary metrics for this specific trip run
    errors_rf = np.abs(pred_rf - actual_soc_arr)
    errors_dt = np.abs(pred_dt - actual_soc_arr)
    errors_knn = np.abs(pred_knn - actual_soc_arr)

    trip_data = {
        "trip_id": trip_id,
        "sample_step_sec": step * 0.1,
        "total_points": len(points),
        "points": points,
        "trip_metrics": {
            "rf": {
                "rmse": round(float(np.sqrt(np.mean((pred_rf - actual_soc_arr)**2))), 4),
                "mae": round(float(np.mean(errors_rf)), 4),
                "max_err": round(float(np.max(errors_rf)), 4)
            },
            "dt": {
                "rmse": round(float(np.sqrt(np.mean((pred_dt - actual_soc_arr)**2))), 4),
                "mae": round(float(np.mean(errors_dt)), 4),
                "max_err": round(float(np.max(errors_dt)), 4)
            },
            "knn": {
                "rmse": round(float(np.sqrt(np.mean((pred_knn - actual_soc_arr)**2))), 4),
                "mae": round(float(np.mean(errors_knn)), 4),
                "max_err": round(float(np.max(errors_knn)), 4)
            }
        }
    }

    TRIP_CACHE[cache_key] = trip_data
    return trip_data


# -------------------------------------------------------------
# HTTP Request Handler
# -------------------------------------------------------------
class BMSDashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=APP_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # API: List test trips
        if path == "/api/trips":
            catalog = get_test_trips_catalog()
            self._send_json(200, {"success": True, "trips": catalog})
            return

        # API: Get single trip trajectory
        if path.startswith("/api/trip/"):
            trip_id = path.replace("/api/trip/", "").strip()
            params = urllib.parse.parse_qs(parsed.query)
            step = int(params.get("step", [5])[0])
            try:
                trip_data = load_and_process_trip(trip_id, step=step)
                self._send_json(200, {"success": True, "data": trip_data})
            except Exception as e:
                self._send_json(404, {"success": False, "error": str(e)})
            return

        # API: Model benchmark table
        if path == "/api/benchmarks":
            benchmarks = {
                "models": [
                    {
                        "model": "Random Forest (Best: 25 Trees)",
                        "architecture": "Ensemble of 25 Decorrelated Decision Trees",
                        "features": "Voltage, Current, Batt Temp, Ambient Temp",
                        "rmse": 5.8876,
                        "mae": 4.3736,
                        "max_error": 26.6956,
                        "inference_time_ms": 1.2,
                        "ram_footprint_mb": 257.7,
                        "status": "Production Selected"
                    },
                    {
                        "model": "Decision Tree (Single)",
                        "architecture": "Single CART Decision Tree (min_samples_leaf=5)",
                        "features": "Voltage, Current, Batt Temp, Ambient Temp",
                        "rmse": 7.0223,
                        "mae": 5.2075,
                        "max_error": 41.7567,
                        "inference_time_ms": 0.08,
                        "ram_footprint_mb": 18.0,
                        "status": "High Variance / Piecewise Jump"
                    },
                    {
                        "model": "k-Nearest Neighbors (k=5)",
                        "architecture": "Instance-based k=5 (StandardScaler, Euclidean)",
                        "features": "Voltage, Current, Batt Temp, Ambient Temp",
                        "rmse": 7.0826,
                        "mae": 5.4040,
                        "max_error": 29.1223,
                        "inference_time_ms": 64.0,
                        "ram_footprint_mb": 24.3,
                        "status": "O(N) MCU Bottleneck"
                    }
                ],
                "battery_spec": {
                    "vehicle": "BMW i3 (94 Ah / 60 Ah)",
                    "nominal_voltage": 350.0,
                    "usable_energy_kwh": 18.8,
                    "chemistry": "LiNiMnCoO2 (NMC)",
                    "nominal_capacity_ah": 60.0
                }
            }
            self._send_json(200, {"success": True, "benchmarks": benchmarks})
            return

        # Fallback to standard static file server
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                req = json.loads(body.decode("utf-8"))
                voltage = float(req.get("voltage", 350.0))
                current = float(req.get("current", 0.0))
                batt_temp = float(req.get("battery_temp", 25.0))
                ambient_temp = float(req.get("ambient_temp", 20.0))

                X = np.array([[voltage, current, batt_temp, ambient_temp]])

                pred_rf = float(MODELS["rf"].predict(X)[0]) if MODELS["rf"] else 0.0
                pred_dt = float(MODELS["dt"].predict(X)[0]) if MODELS["dt"] else 0.0

                if MODELS["knn"] and MODELS["knn_scaler"]:
                    X_scaled = MODELS["knn_scaler"].transform(X)
                    pred_knn = float(MODELS["knn"].predict(X_scaled)[0])
                else:
                    pred_knn = 0.0

                # Clamping bounds for physical battery safety
                pred_rf = max(0.0, min(100.0, pred_rf))
                pred_dt = max(0.0, min(100.0, pred_dt))
                pred_knn = max(0.0, min(100.0, pred_knn))

                # Estimated Driving Range (BMW i3 ~ 150 km nominal at 100% SOC)
                range_km = round((pred_rf / 100.0) * 150.0, 1)
                power_kw = round((voltage * current) / 1000.0, 2)

                self._send_json(200, {
                    "success": True,
                    "inputs": {
                        "voltage": voltage,
                        "current": current,
                        "battery_temp": batt_temp,
                        "ambient_temp": ambient_temp
                    },
                    "predictions": {
                        "rf": round(pred_rf, 2),
                        "dt": round(pred_dt, 2),
                        "knn": round(pred_knn, 2)
                    },
                    "telemetry": {
                        "estimated_range_km": range_km,
                        "power_kw": power_kw,
                        "is_regenerating": current > 0.5
                    }
                })
            except Exception as e:
                self._send_json(400, {"success": False, "error": str(e)})
            return

        self._send_json(404, {"success": False, "error": "Not Found"})

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Mute repetitive static asset logs, show only API calls
        try:
            msg = format % args
            if "/api/" in msg:
                sys.stdout.write(f"[API] {self.address_string()} - {msg}\n")
        except Exception:
            pass


# -------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------
class BMSDashboardServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def run_server(port: int = 8000):
    load_models()
    server_address = ("", port)
    httpd = BMSDashboardServer(server_address, BMSDashboardHandler)
    print("\n" + "=" * 60)
    print(f"  BMS Digital Twin Server running at: http://localhost:{port}")
    print(f"  Open http://localhost:{port} in your browser.")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
