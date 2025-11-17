# main.py — Rolling walk-forward simulation (7-day window) with 3-model ensemble
# -------------------------------------------------------------------------------
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

# ------------------------
# 5) API endpoint: /simulate
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
            "simulate": "POST /simulate (optional query param max_days)",
            "simulateTomorrow": "POST /simulateTomorrow?currentDateTime=YYYY-MM-DD",
            "simulateTomorrowAuto": "POST /simulateTomorrowAuto (uses today's date automatically)"
        },
        "notes": "Simulation aggregates multiple transactions per day into daily totals."
    }