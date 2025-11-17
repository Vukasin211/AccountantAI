from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# ------------------------
# 1) App + CORS
# ------------------------
app = FastAPI(title="Expense Predictor - Rolling Simulation")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# 2) Config
# ------------------------
CSV_FILE = "expenses4.csv"         
WINDOW_SIZE = 11                    
PCT_THRESHOLD = 0.30               
MODEL_FILES = {
    "low":  "lstm_expense_model_low.h5",
    "mid":  "lstm_expense_model_mid.h5",
    "high": "lstm_expense_model_high.h5"
}

# ------------------------
# 3) Load CSV and models at startup
# ------------------------
print("Loading CSV and models...")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found in working directory.")

_raw_df = pd.read_csv(CSV_FILE)
if "Type" not in _raw_df.columns or "Amount" not in _raw_df.columns or "Date" not in _raw_df.columns:
    raise ValueError("CSV must contain columns: Date, Amount, Type (at least).")

# Keep only expense rows
df = _raw_df[_raw_df['Type'].str.lower() == 'expense'].copy()
df['Date'] = pd.to_datetime(df['Date'], dayfirst=False, errors='coerce')
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

if df.empty:
    raise ValueError("No expense rows found in CSV after filtering Type == 'Expense'.")

# --- Aggregate by day ---
df_daily = df.groupby('Date', as_index=False)['Amount'].sum()
dates_daily = df_daily['Date'].dt.to_pydatetime().tolist()
amounts_daily = df_daily['Amount'].values.astype(float)

# scaler fitted on all daily amounts
scaler = StandardScaler()
scaler.fit(df_daily[['Amount']])

# thresholds
low_th = df_daily['Amount'].quantile(0.33)
high_th = df_daily['Amount'].quantile(0.66)

# load models
models = {}
for label, path in MODEL_FILES.items():
    if os.path.exists(path):
        models[label] = load_model(path, compile=False)
        print(f"Loaded model: {path}")
    else:
        print(f"Model not found (skipping): {path}")

if not models:
    raise RuntimeError("No models loaded. Place model files in working directory.")

# ------------------------
# 4) Utility helpers
# ------------------------
def choose_model_by_value(value_real):
    if value_real <= low_th and 'low' in models:
        return 'low', models['low']
    if value_real <= high_th and 'mid' in models:
        return 'mid', models['mid']
    if 'high' in models:
        return 'high', models['high']
    return next(iter(models.items()))

def seq_to_model_input(seq_scaled):
    return np.array(seq_scaled).reshape(1, len(seq_scaled), 1)

def inverse_amount(scaled_val):
    df_inv = pd.DataFrame([[scaled_val]], columns=["Amount"])
    return float(scaler.inverse_transform(df_inv)[0][0])

def scale_amounts(arr_amounts):
    df_in = pd.DataFrame(np.array(arr_amounts).reshape(-1,1), columns=["Amount"])
    scaled = scaler.transform(df_in)
    return scaled.reshape(-1)

def plot_results(dates, actuals, preds, correct_mask, overall_accuracy):
    N = len(actuals)
    plt.figure(figsize=(12,6))
    plt.plot(dates, actuals, label='Actual', marker='o', linewidth=1)
    plt.plot(dates, preds, label='Predicted', marker='o', linestyle='dashed', linewidth=1)
    for i in range(N):
        plt.axvspan(dates[i]-timedelta(hours=12), dates[i]+timedelta(hours=12), color='green' if correct_mask[i] else 'red', alpha=0.08)
    plt.xticks(rotation=45)
    plt.title(f"Rolling predictions (window={WINDOW_SIZE}) — overall correct (±{int(PCT_THRESHOLD*100)}%): {overall_accuracy:.2f}%")
    plt.xlabel("Date")
    plt.ylabel("Daily Expense Amount")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return buf

# -------------------------------
# Individual Metric Calculation Functions (Single Return)
# -------------------------------

def get_valid_prediction_pairs(y_pred, y_true):
    """Filter to only pairs where actual value exists and values are finite"""
    valid_preds = []
    valid_actuals = []
    
    for pred, actual in zip(y_pred, y_true):
        if actual is not None and np.isfinite(pred) and np.isfinite(actual):
            valid_preds.append(pred)
            valid_actuals.append(actual)
    
    return valid_preds, valid_actuals

def get_all_predictions(y_pred, y_true):
    """Get all predictions regardless of whether actual exists"""
    all_preds = []
    all_actuals = []
    
    for pred, actual in zip(y_pred, y_true):
        if np.isfinite(pred):
            all_preds.append(pred)
            all_actuals.append(actual)  # actual can be None
    
    return all_preds, all_actuals

def calculate_mae_single(y_true, y_pred):
    """Calculate Mean Absolute Error - returns single float"""
    valid_preds, valid_actuals = get_valid_prediction_pairs(y_pred, y_true)
    if len(valid_actuals) == 0:
        return 0.0
    try:
        mae = float(mean_absolute_error(valid_actuals, valid_preds))
        return mae if np.isfinite(mae) else 0.0
    except:
        return 0.0

def calculate_rmse_single(y_true, y_pred):
    """Calculate Root Mean Squared Error - returns single float"""
    valid_preds, valid_actuals = get_valid_prediction_pairs(y_pred, y_true)
    if len(valid_actuals) == 0:
        return 0.0
    try:
        rmse = float(np.sqrt(mean_squared_error(valid_actuals, valid_preds)))
        return rmse if np.isfinite(rmse) else 0.0
    except:
        return 0.0

def calculate_r2_single(y_true, y_pred):
    """Calculate R² Score - returns single float"""
    valid_preds, valid_actuals = get_valid_prediction_pairs(y_pred, y_true)
    if len(valid_actuals) < 2:  # Need at least 2 points for R²
        return 0.0
    try:
        r2 = float(r2_score(valid_actuals, valid_preds))
        return r2 if np.isfinite(r2) else 0.0
    except:
        return 0.0

def calculate_accuracy_percentage_single(y_true, y_pred, threshold=PCT_THRESHOLD):
    """Calculate accuracy percentage - returns single float"""
    valid_preds, valid_actuals = get_valid_prediction_pairs(y_pred, y_true)
    if len(valid_actuals) == 0:
        return 0.0
    
    try:
        correct_mask = []
        for pred, actual in zip(valid_preds, valid_actuals):
            if actual == 0:
                correct_mask.append(True)
            else:
                error_ratio = abs(pred - actual) / actual
                correct_mask.append(error_ratio <= threshold)
        
        correct_count = sum(correct_mask)
        accuracy = (correct_count / len(valid_actuals) * 100) if len(valid_actuals) > 0 else 0.0
        return float(accuracy) if np.isfinite(accuracy) else 0.0
    except:
        return 0.0

def calculate_correct_predictions_single(y_true, y_pred, threshold=PCT_THRESHOLD):
    """Calculate number of correct predictions - returns single integer"""
    valid_preds, valid_actuals = get_valid_prediction_pairs(y_pred, y_true)
    if len(valid_actuals) == 0:
        return 0
    
    try:
        correct_mask = []
        for pred, actual in zip(valid_preds, valid_actuals):
            if actual == 0:
                correct_mask.append(True)
            else:
                error_ratio = abs(pred - actual) / actual
                correct_mask.append(error_ratio <= threshold)
        
        return int(sum(correct_mask))
    except:
        return 0

def calculate_total_predictions_single(y_true, y_pred):
    """Calculate total valid predictions - returns single integer"""
    valid_preds, valid_actuals = get_valid_prediction_pairs(y_pred, y_true)
    return len(valid_actuals)

def calculate_total_all_predictions_single(y_true, y_pred):
    """Calculate total ALL predictions (including future ones without actuals) - returns single integer"""
    all_preds, all_actuals = get_all_predictions(y_pred, y_true)
    return len(all_preds)

# ------------------------
# 5) API endpoints
# ------------------------

@app.post("/simulateTomorrow")
def simulateTomorrow(currentDateTime: str = Query(..., description="Current date in YYYY-MM-DD format")):
    """
    Predict tomorrow's expense based on current date and historical data
    """
    try:
        # Parse the current date
        current_date = datetime.strptime(currentDateTime, "%Y-%m-%d")
        tomorrow_date = current_date + timedelta(days=1)
        
        # Check if we have enough historical data
        if len(amounts_daily) < WINDOW_SIZE:
            return JSONResponse({
                "error": f"Not enough historical data. Need at least {WINDOW_SIZE} days, but have {len(amounts_daily)}"
            }, status_code=400)
        
        # Get the most recent WINDOW_SIZE days of data
        recent_data = amounts_daily[-WINDOW_SIZE:]
        recent_dates = dates_daily[-WINDOW_SIZE:]
        
        # Find today's spending in the data
        today_spending = None
        today_date_str = current_date.strftime("%Y-%m-%d")
        
        # Look for today's date in the historical data
        for i, date in enumerate(recent_dates):
            if date.strftime("%Y-%m-%d") == today_date_str:
                today_spending = float(recent_data[i])
                break
        
        # If today's data not found in recent window, check the entire dataset
        if today_spending is None:
            for i, date in enumerate(dates_daily):
                if date.strftime("%Y-%m-%d") == today_date_str:
                    today_spending = float(amounts_daily[i])
                    break
        
        # Scale the recent data
        scaled_recent = scale_amounts(recent_data)
        
        # Use the most recent actual amount to choose the model
        last_real_amount = float(recent_data[-1])
        model_label, model_obj = choose_model_by_value(last_real_amount)
        
        # Prepare input for prediction
        X = seq_to_model_input(scaled_recent)
        
        # Make prediction
        pred_scaled = float(model_obj.predict(X, verbose=0)[0][0])
        pred_amount = inverse_amount(pred_scaled)
        
        # Get some context about recent spending
        recent_avg = float(np.mean(recent_data))
        recent_std = float(np.std(recent_data))
        
        # Determine spending trend
        if len(recent_data) >= 3:
            last_three = recent_data[-3:]
            trend = "increasing" if last_three[-1] > last_three[0] else "decreasing" if last_three[-1] < last_three[0] else "stable"
        else:
            trend = "unknown"
        
        # Prepare response
        response = {
            "prediction": {
                "date": tomorrow_date.strftime("%Y-%m-%d"),
                "predicted_amount": round(pred_amount, 2),
                "model_used": model_label,
                "confidence_interval": {
                    "lower_bound": round(max(0, pred_amount - recent_std), 2),
                    "upper_bound": round(pred_amount + recent_std, 2)
                }
            },
            "current_day": {
                "date": today_date_str,
                "actual_amount": round(today_spending, 2) if today_spending is not None else None,
                "status": "found" if today_spending is not None else "not_found_in_data"
            },
            "context": {
                "recent_average": round(recent_avg, 2),
                "recent_volatility": round(recent_std, 2),
                "trend": trend,
                "last_actual_amount": round(last_real_amount, 2),
                "days_used_for_prediction": WINDOW_SIZE
            },
            "model_thresholds": {
                "low_threshold": round(low_th, 2),
                "high_threshold": round(high_th, 2)
            }
        }
        
        return response
        
    except ValueError as e:
        return JSONResponse({
            "error": f"Invalid date format. Please use YYYY-MM-DD format. Error: {str(e)}"
        }, status_code=400)
    except Exception as e:
        return JSONResponse({
            "error": f"Prediction failed: {str(e)}"
        }, status_code=500)

@app.post("/simulateTomorrowAuto")
def simulateTomorrowAuto():
    """
    Predict tomorrow's expense using today's date automatically
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return simulateTomorrow(today)

@app.post("/simulateData")
def simulateData(max_days: int = Query(None, description="How many sequential days to simulate")):
    """
    Rolling simulation that forces predictions for sequential calendar days.
    First prediction day = tomorrow from PC date.
    If the CSV does not contain that date, actual = None.
    """
    
    # Must have enough history
    if len(amounts_daily) < WINDOW_SIZE:
        return JSONResponse(
            {"error": f"Not enough data. Need {WINDOW_SIZE} days."},
            status_code=400
        )

    # -------------------------------
    # 1) Detect today from PC
    # -------------------------------
    today = datetime.now().date()
    first_pred_date = today + timedelta(days=1)

    # If user specifies max_days → limit
    if max_days is None or max_days <= 0:
        max_days = 30  # default simulate 30 days

    # -------------------------------
    # 2) Prepare a lookup dict of real CSV amounts
    # -------------------------------
    real_lookup = {
        d.date(): float(a) for d, a in zip(dates_daily, amounts_daily)
    }

    # -------------------------------
    # 3) Start with last WINDOW_SIZE real values
    # -------------------------------
    recent_values = list(amounts_daily[-WINDOW_SIZE:])
    recent_scaled = scale_amounts(recent_values)

    preds, actuals, datelist = [], [], []

    # -------------------------------
    # 4) Sequential day-by-day prediction loop
    # -------------------------------
    for step in range(max_days):

        current_pred_date = first_pred_date + timedelta(days=step)
        datelist.append(current_pred_date.strftime("%Y-%m-%d"))

        # Select model based on last real known value
        last_real = recent_values[-1]
        model_label, model_obj = choose_model_by_value(last_real)

        # Prepare input
        X = seq_to_model_input(recent_scaled)
        pred_scaled = float(model_obj.predict(X, verbose=0)[0][0])
        pred_amount = inverse_amount(pred_scaled)

        # Ensure prediction is finite
        if not np.isfinite(pred_amount):
            pred_amount = 0.0

        preds.append(pred_amount)

        # Check if this date exists in real CSV
        actual = real_lookup.get(current_pred_date, None)
        actuals.append(actual)

        # If real data exists → use real value in window
        if actual is not None:
            next_val = actual
        else:
            # No real data → treat prediction as next known history
            next_val = pred_amount

        # Update rolling window
        recent_values.append(next_val)
        recent_values = recent_values[-WINDOW_SIZE:]
        recent_scaled = scale_amounts(recent_values)

    # -------------------------------
    # 5) Build JSON response using individual metric functions
    # -------------------------------
    results = []
    
    for d, p, a in zip(datelist, preds, actuals):
        if a is not None:
            within = abs(p - a) / a <= PCT_THRESHOLD if a != 0 else True
        else:
            within = None

        results.append({
            "date": d,
            "predicted": round(p, 2),
            "actual": round(a, 2) if a is not None else None,
            "within_threshold": within
        })

    # CALCULATE EACH METRIC USING INDIVIDUAL FUNCTIONS
    mae = calculate_mae_single(actuals, preds)
    rmse = calculate_rmse_single(actuals, preds)
    r2 = calculate_r2_single(actuals, preds)
    accuracy_pct = calculate_accuracy_percentage_single(actuals, preds)
    correct_count = calculate_correct_predictions_single(actuals, preds)
    
    # Use the NEW function that counts ALL predictions, not just validated ones
    total_count = calculate_total_all_predictions_single(actuals, preds)

    # Ensure all metrics are JSON serializable
    response_data = {
        "auto_detected_today": today.strftime("%Y-%m-%d"),
        "simulation_start_day": first_pred_date.strftime("%Y-%m-%d"),
        "overall_metrics": {
            "mean_absolute_error": float(mae) if np.isfinite(mae) else 0.0,
            "root_mean_squared_error": float(rmse) if np.isfinite(rmse) else 0.0,
            "r2_score": float(r2) if np.isfinite(r2) else 0.0,
            "accuracy_percentage": float(accuracy_pct) if np.isfinite(accuracy_pct) else 0.0,
            "correct_predictions": int(correct_count),
            "total_predictions": int(total_count),
            "threshold": f"±{int(PCT_THRESHOLD*100)}%"
        },
        "predictions": results
    }
    
    return response_data

@app.post("/simulate")
def simulate(max_days: int = Query(None, description="Limit how many days to simulate (default = all)")):
    if len(amounts_daily) < WINDOW_SIZE + 1:
        return JSONResponse({"error": f"Need at least {WINDOW_SIZE+1} daily rows to simulate."}, status_code=400)

    scaled_all = scale_amounts(amounts_daily)
    start_idx = WINDOW_SIZE
    end_idx = len(amounts_daily)
    if max_days is not None and max_days > 0:
        end_idx = min(end_idx, start_idx + max_days)

    preds_list, actuals_list, pred_dates, correct_mask = [], [], [], []

    for i in range(start_idx, end_idx):
        window_scaled = scaled_all[i-WINDOW_SIZE:i]
        last_real_amount = float(amounts_daily[i-1])
        model_label, model_obj = choose_model_by_value(last_real_amount)
        X = seq_to_model_input(window_scaled)
        pred_scaled = float(model_obj.predict(X, verbose=0)[0][0])
        pred_amount = inverse_amount(pred_scaled)
        actual_amount = float(amounts_daily[i])
        pred_date = dates_daily[i]

        preds_list.append(pred_amount)
        actuals_list.append(actual_amount)
        pred_dates.append(pred_date)
        correct_mask.append(abs(pred_amount-actual_amount)/actual_amount <= PCT_THRESHOLD if actual_amount!=0 else True)

    y_true = np.array(actuals_list)
    y_pred = np.array(preds_list)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    pct_within = float(np.mean(correct_mask) * 100)

    buf = plot_results(pred_dates, y_true, y_pred, correct_mask, pct_within)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# ------------------------
# 6) Root
# ------------------------
@app.get("/")
def root():
    return {
        "message": "Rolling simulation ready.",
        "endpoints": {
            "simulate": "POST /simulate (optional query param max_days) - returns PNG plot",
            "simulateData": "POST /simulateData (optional query param max_days) - returns JSON data", 
            "simulateTomorrow": "POST /simulateTomorrow?currentDateTime=YYYY-MM-DD - predict tomorrow",
            "simulateTomorrowAuto": "POST /simulateTomorrowAuto - predict tomorrow using today's date"
        },
        "notes": "Simulation aggregates multiple transactions per day into daily totals.",
        "current_data_stats": {
            "total_days": len(amounts_daily),
            "date_range": {
                "start": dates_daily[0].strftime("%Y-%m-%d") if dates_daily else "No data",
                "end": dates_daily[-1].strftime("%Y-%m-%d") if dates_daily else "No data"
            },
            "window_size": WINDOW_SIZE,
            "accuracy_threshold": f"±{int(PCT_THRESHOLD*100)}%"
        }
    }