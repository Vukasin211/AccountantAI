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
CSV_FILE = "expenses5.csv"         
WINDOW_SIZE = 7                    
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

def calculate_date_features(date):
    """Calculate date features for LSTM input"""
    day_of_month = date.day
    month = date.month
    
    day_sin = np.sin(2 * np.pi * day_of_month / 31)
    day_cos = np.cos(2 * np.pi * day_of_month / 31)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    return day_sin, day_cos, month_sin, month_cos

def seq_to_model_input(seq_scaled, dates):
    """Create 5-feature input with actual date features for LSTM"""
    seq_5d = []
    for i, (amount, date) in enumerate(zip(seq_scaled, dates)):
        # Calculate date features for each date in the sequence
        day_sin, day_cos, month_sin, month_cos = calculate_date_features(date)
        seq_5d.append([amount, day_sin, day_cos, month_sin, month_cos])
    
    return np.array(seq_5d).reshape(1, len(seq_scaled), 5)

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
    
    # Plot actual values (including zeros/missing as 0)
    actuals_to_plot = [0 if a is None or np.isnan(a) else a for a in actuals]
    plt.plot(dates, actuals_to_plot, label='Actual', marker='o', linewidth=1)
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
def simulateTomorrow(
    currentDateTime: str = Query(..., description="Current date in YYYY-MM-DD format"),
    accuracy: float = Query(0.30, description="Accuracy threshold as decimal (default 0.30 = 30%)")
):
    """
    Predict tomorrow's expense based on current date and historical data
    """
    try:
        # Validate accuracy parameter
        if accuracy <= 0 or accuracy > 1:
            return JSONResponse({
                "error": "Accuracy parameter must be between 0 and 1 (e.g., 0.30 for 30%)"
            }, status_code=400)
            
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
        
        # Prepare input for prediction (with date features)
        X = seq_to_model_input(scaled_recent, recent_dates)
        
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
            },
            "accuracy_threshold_used": accuracy
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
def simulateTomorrowAuto(accuracy: float = Query(0.30, description="Accuracy threshold as decimal (default 0.30 = 30%)")):
    """
    Predict tomorrow's expense using today's date automatically
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return simulateTomorrow(today, accuracy)

@app.post("/simulate")
def simulate(
    max_days: int = Query(None, description="Limit how many days to simulate (default = 30)"),
    accuracy: float = Query(0.30, description="Accuracy threshold as decimal (default 0.30 = 30%)")
):
    """
    Rolling future simulation starting from TOMORROW (PC date + 1).
    Includes detailed console logging for debugging.
    """

    if len(amounts_daily) < WINDOW_SIZE:
        return JSONResponse(
            {"error": f"Need at least {WINDOW_SIZE} daily rows to simulate."},
            status_code=400
        )

    # Validate accuracy parameter
    if accuracy <= 0 or accuracy > 1:
        return JSONResponse({
            "error": "Accuracy parameter must be between 0 and 1 (e.g., 0.30 for 30%)"
        }, status_code=400)

    # -------------------------------
    # 1) Detect today + 1 (first prediction day)
    # -------------------------------
    today = datetime.now().date()
    first_pred_date = today + timedelta(days=1)

    if max_days is None or max_days <= 0:
        max_days = 30

    print("\n=== SIMULATION DEBUG START ===")
    print(f"PC Today: {today}")
    print(f"Prediction Start Day: {first_pred_date}")
    print(f"Simulating {max_days} days")
    print(f"Accuracy threshold: {accuracy*100}%")
    print()

    # -------------------------------
    # 2) Real data lookup
    # -------------------------------
    real_lookup = {d.date(): float(a) for d, a in zip(dates_daily, amounts_daily)}

    # -------------------------------
    # 3) Seed last WINDOW_SIZE values
    # -------------------------------
    recent_values = list(amounts_daily[-WINDOW_SIZE:])
    recent_dates = list(dates_daily[-WINDOW_SIZE:])
    recent_scaled = scale_amounts(recent_values)

    print(f"Seed window ({WINDOW_SIZE} days): {recent_values}")
    print(f"CSV range: {dates_daily[0].date()} → {dates_daily[-1].date()}\n")

    preds, actuals, pred_dates = [], [], []

    # -------------------------------
    # 4) Day-by-day prediction loop
    # -------------------------------
    for step in range(max_days):

        current_pred_date = first_pred_date + timedelta(days=step)
        pred_dates.append(current_pred_date)

        print(f"# Day {step+1} → {current_pred_date}")

        # -------- Model selection --------
        last_real_for_choice = recent_values[-1]
        model_label, model_obj = choose_model_by_value(last_real_for_choice)
        print(f"  Model used: {model_label} (last real={last_real_for_choice})")

        # -------- Prediction --------
        X = seq_to_model_input(recent_scaled, recent_dates)
        pred_scaled = float(model_obj.predict(X, verbose=0)[0][0])
        pred_amount = inverse_amount(pred_scaled)

        if not np.isfinite(pred_amount):
            print("  WARNING: Non-finite prediction → forcing to 0")
            pred_amount = 0.0

        preds.append(pred_amount)
        print(f"  Predicted amount: {pred_amount:.2f}")

        # -------- Actual lookup --------
        actual_amount = real_lookup.get(current_pred_date, None)
        actuals.append(actual_amount)

        if actual_amount is None:
            print("  Actual: MISSING from CSV")
        else:
            print(f"  Actual (CSV): {actual_amount:.2f}")

        # -------- Update rolling window --------
        value_for_next_window = actual_amount if actual_amount is not None else pred_amount

        recent_values.append(value_for_next_window)
        recent_values = recent_values[-WINDOW_SIZE:]

        # Update dates window
        recent_dates.append(current_pred_date)
        recent_dates = recent_dates[-WINDOW_SIZE:]

        # Rescale after update
        recent_scaled = scale_amounts(recent_values)

        print(f"  Next value used for LSTM window: {value_for_next_window:.2f}")
        print(f"  Updated window tail: {recent_values[-3:]}\n")

    print("=== SIMULATION DEBUG END ===\n")

    # -------------------------------
    # 5) Correctness mask using the provided accuracy parameter
    # -------------------------------
    correctness = []
    for p, a in zip(preds, actuals):
        if a is None:
            correctness.append(False)
        elif a == 0:
            correctness.append(True)
        else:
            correctness.append(abs(p - a) / a <= accuracy)

    valid_mask = [c for c, a in zip(correctness, actuals) if a is not None]
    overall_accuracy = float(np.mean(valid_mask) * 100) if valid_mask else 0.0

    # -------------------------------
    # 6) Plot result - plot zeros for missing dates
    # -------------------------------
    # Replace None values with 0 for plotting
    actuals_clean = [0 if a is None else a for a in actuals]
    buf = plot_results(pred_dates, actuals_clean, preds, correctness, overall_accuracy)
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
            "simulate": "POST /simulate (optional query params max_days, accuracy) - returns PNG plot",
            "simulateTomorrow": "POST /simulateTomorrow?currentDateTime=YYYY-MM-DD&accuracy=0.30 - predict tomorrow",
            "simulateTomorrowAuto": "POST /simulateTomorrowAuto?accuracy=0.30 - predict tomorrow using today's date"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)