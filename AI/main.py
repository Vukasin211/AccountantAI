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
CSV_FILE = "Personal_Finance_Dataset2.csv"         
WINDOW_SIZE = 7                    
PCT_THRESHOLD = 0.30               
MODEL_FILES = {
    "low":  "lstm_expense_model_low.h5",
    "mid":  "lstm_expense_model_mid.h5",
    "high": "lstm_expense_model_high.h5"
}
SINGLE_MODEL_FILE = "lstm_expense_model_single.h5"  # Single model file

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

# load multi-models
models = {}
for label, path in MODEL_FILES.items():
    if os.path.exists(path):
        models[label] = load_model(path, compile=False)
        print(f"Loaded model: {path}")
    else:
        print(f"Model not found (skipping): {path}")

# load single model
single_model = None
if os.path.exists(SINGLE_MODEL_FILE):
    single_model = load_model(SINGLE_MODEL_FILE, compile=False)
    print(f"Loaded single model: {SINGLE_MODEL_FILE}")
else:
    print(f"Single model not found: {SINGLE_MODEL_FILE}")

if not models and single_model is None:
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
    """Create 1-feature input for LSTM (compatible with trained models)"""
    # Use only the amount values - models were trained with 1 feature
    return np.array(seq_scaled).reshape(1, len(seq_scaled), 1)

def inverse_amount(scaled_val):
    df_inv = pd.DataFrame([[scaled_val]], columns=["Amount"])
    return float(scaler.inverse_transform(df_inv)[0][0])

def scale_amounts(arr_amounts):
    df_in = pd.DataFrame(np.array(arr_amounts).reshape(-1,1), columns=["Amount"])
    scaled = scaler.transform(df_in)
    return scaled.reshape(-1)

def plot_results(dates, actuals, preds, correct_mask, overall_accuracy, accuracy_threshold=PCT_THRESHOLD, overestimate_ok=False, single_model=False):
    N = len(actuals)
    plt.figure(figsize=(12,6))
    
    # Plot actual values (including zeros/missing as 0)
    actuals_to_plot = [0 if a is None or np.isnan(a) else a for a in actuals]
    plt.plot(dates, actuals_to_plot, label='Actual', marker='o', linewidth=1)
    plt.plot(dates, preds, label='Predicted', marker='o', linestyle='dashed', linewidth=1)
    
    for i in range(N):
        plt.axvspan(dates[i]-timedelta(hours=12), dates[i]+timedelta(hours=12), color='green' if correct_mask[i] else 'red', alpha=0.08)
    
    plt.xticks(rotation=45)
    
    # Update title based on overestimate_ok and single_model settings
    model_type = "Single Model" if single_model else "Multi-Model"
    if overestimate_ok:
        plt.title(f"Rolling predictions ({model_type}, window={WINDOW_SIZE}) — overall correct (±{int(accuracy_threshold*100)}% or overestimate): {overall_accuracy:.2f}%")
    else:
        plt.title(f"Rolling predictions ({model_type}, window={WINDOW_SIZE}) — overall correct (±{int(accuracy_threshold*100)}%): {overall_accuracy:.2f}%")
    
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

def calculate_accuracy_percentage_single(y_true, y_pred, threshold=PCT_THRESHOLD, overestimate_ok=False):
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
                if overestimate_ok:
                    # Consider correct if within threshold OR prediction is higher than actual
                    error_ratio = abs(pred - actual) / actual
                    correct_mask.append(error_ratio <= threshold or pred >= actual)
                else:
                    # Original logic - only within threshold
                    error_ratio = abs(pred - actual) / actual
                    correct_mask.append(error_ratio <= threshold)
        
        correct_count = sum(correct_mask)
        accuracy = (correct_count / len(valid_actuals) * 100) if len(valid_actuals) > 0 else 0.0
        return float(accuracy) if np.isfinite(accuracy) else 0.0
    except:
        return 0.0

def calculate_correct_predictions_single(y_true, y_pred, threshold=PCT_THRESHOLD, overestimate_ok=False):
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
                if overestimate_ok:
                    # Consider correct if within threshold OR prediction is higher than actual
                    error_ratio = abs(pred - actual) / actual
                    correct_mask.append(error_ratio <= threshold or pred >= actual)
                else:
                    # Original logic - only within threshold
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
# 5) API endpoints - Multi-Model (Original)
# ------------------------

@app.post("/simulateTomorrow")
def simulateTomorrow(
    currentDateTime: str = Query(..., description="Current date in YYYY-MM-DD format"),
    accuracy: float = Query(0.30, description="Accuracy threshold as decimal (default 0.30 = 30%)")
):
    """
    Predict tomorrow's expense based on current date and historical data (Multi-Model)
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
            "accuracy_threshold_used": accuracy,
            "model_type": "multi-model"
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
    Predict tomorrow's expense using today's date automatically (Multi-Model)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return simulateTomorrow(today, accuracy)

@app.post("/simulate")
def simulate(
    max_days: int = Query(None, description="Limit how many days to simulate (default = 30)"),
    accuracy: float = Query(0.30, description="Accuracy threshold as decimal (default 0.30 = 30%)"),
    overestimate_ok: int = Query(0, description="If set to 1, considers overestimates as correct (default 0)")
):
    """
    Rolling future simulation starting from TOMORROW (PC date + 1). (Multi-Model)
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

    # Validate overestimate_ok parameter
    if overestimate_ok not in [0, 1]:
        return JSONResponse({
            "error": "overestimate_ok parameter must be 0 or 1"
        }, status_code=400)

    # Convert to boolean
    overestimate_ok_bool = bool(overestimate_ok)

    # -------------------------------
    # 1) Detect today + 1 (first prediction day)
    # -------------------------------
    today = datetime.now().date()
    first_pred_date = today + timedelta(days=1)

    if max_days is None or max_days <= 0:
        max_days = 30

    print("\n=== SIMULATION DEBUG START (Multi-Model) ===")
    print(f"PC Today: {today}")
    print(f"Prediction Start Day: {first_pred_date}")
    print(f"Simulating {max_days} days")
    print(f"Accuracy threshold: {accuracy*100}%")
    print(f"Overestimate considered correct: {overestimate_ok_bool}")
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
    # 5) Correctness mask using the provided accuracy parameter and overestimate setting
    # -------------------------------
    correctness = []
    for p, a in zip(preds, actuals):
        if a is None:
            correctness.append(False)
        elif a == 0:
            correctness.append(True)
        else:
            if overestimate_ok_bool:
                # Consider correct if within accuracy threshold OR prediction is higher than actual
                error_ratio = abs(p - a) / a
                correctness.append(error_ratio <= accuracy or p >= a)
            else:
                # Original logic - only within accuracy threshold
                correctness.append(abs(p - a) / a <= accuracy)

    valid_mask = [c for c, a in zip(correctness, actuals) if a is not None]
    overall_accuracy = float(np.mean(valid_mask) * 100) if valid_mask else 0.0

    # -------------------------------
    # 6) Plot result - plot zeros for missing dates
    # -------------------------------
    # Replace None values with 0 for plotting
    actuals_clean = [0 if a is None else a for a in actuals]
    buf = plot_results(pred_dates, actuals_clean, preds, correctness, overall_accuracy, accuracy, overestimate_ok_bool, single_model=False)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


#----------------------------------------------------------------------------


@app.get("/simulate-json")
def simulate_json(
    max_days: int = Query(None, description="Limit how many days to simulate (default = 30)"),
    accuracy: float = Query(0.30, description="Accuracy threshold as decimal (default 0.30 = 30%)"),
    overestimate_ok: int = Query(0, description="If set to 1, considers overestimates as correct (default 0)")
):
    """
    Rolling future simulation starting from TOMORROW (PC date + 1) in JSON format.
    Includes comprehensive accuracy statistics.
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

    # Validate overestimate_ok parameter
    if overestimate_ok not in [0, 1]:
        return JSONResponse({
            "error": "overestimate_ok parameter must be 0 or 1"
        }, status_code=400)

    # Convert to boolean
    overestimate_ok_bool = bool(overestimate_ok)

    # -------------------------------
    # 1) Detect today + 1 (first prediction day)
    # -------------------------------
    today = datetime.now().date()
    first_pred_date = today + timedelta(days=1)

    if max_days is None or max_days <= 0:
        max_days = 30

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

    pred_dates = []
    predictions = []
    actuals = []
    model_used_list = []
    window_tails = []
    correctness_list = []
    errors = []
    absolute_errors = []
    squared_errors = []
    percentage_errors = []

    # -------------------------------
    # 4) Day-by-day prediction loop
    # -------------------------------
    for step in range(max_days):
        current_pred_date = first_pred_date + timedelta(days=step)
        pred_dates.append(current_pred_date)

        # -------- Model selection --------
        last_real_for_choice = recent_values[-1]
        model_label, model_obj = choose_model_by_value(last_real_for_choice)

        # -------- Prediction --------
        X = seq_to_model_input(recent_scaled, recent_dates)
        pred_scaled = float(model_obj.predict(X, verbose=0)[0][0])
        pred_amount = float(inverse_amount(pred_scaled))

        if not np.isfinite(pred_amount):
            pred_amount = 0.0

        predictions.append(pred_amount)

        # -------- Actual lookup --------
        actual_amount = real_lookup.get(current_pred_date, None)
        actual_amount_float = float(actual_amount) if actual_amount is not None else None
        actuals.append(actual_amount_float)

        # -------- Calculate error metrics --------
        error = None
        abs_error = None
        sq_error = None
        perc_error = None
        error_ratio = None
        correct = False
        
        if actual_amount is not None:
            error = float(pred_amount - actual_amount)
            abs_error = float(abs(error))
            sq_error = float(error ** 2)
            perc_error = float((abs_error / actual_amount) * 100) if actual_amount != 0 else 0.0
            error_ratio = float(abs_error / actual_amount) if actual_amount != 0 else 0.0
            
            errors.append(error)
            absolute_errors.append(abs_error)
            squared_errors.append(sq_error)
            percentage_errors.append(perc_error)
            
            # Determine correctness based on accuracy threshold
            if actual_amount == 0:
                correct = True
            else:
                if overestimate_ok_bool:
                    # Consider correct if within accuracy threshold OR prediction is higher than actual
                    correct = error_ratio <= accuracy or pred_amount >= actual_amount
                else:
                    # Original logic - only within accuracy threshold
                    correct = error_ratio <= accuracy
        
        correctness_list.append(correct)

        # -------- Update rolling window --------
        value_for_next_window = actual_amount if actual_amount is not None else pred_amount

        recent_values.append(value_for_next_window)
        recent_values = recent_values[-WINDOW_SIZE:]

        # Update dates window
        recent_dates.append(current_pred_date)
        recent_dates = recent_dates[-WINDOW_SIZE:]

        # Rescale after update
        recent_scaled = scale_amounts(recent_values)

        # -------- Store window tail for debugging --------
        window_tails.append([float(x) for x in recent_values[-3:]])

        # -------- Store model info --------
        model_used_list.append(model_label)

    # -------------------------------
    # 5) Calculate comprehensive statistics
    # -------------------------------
    # Filter only days with actual values for statistics
    valid_indices = [i for i, a in enumerate(actuals) if a is not None]
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_actuals = [actuals[i] for i in valid_indices]
    valid_errors = errors if errors else []
    valid_absolute_errors = absolute_errors if absolute_errors else []
    valid_squared_errors = squared_errors if squared_errors else []
    valid_percentage_errors = percentage_errors if percentage_errors else []
    
    # Overall accuracy (based on threshold)
    valid_correctness = [c for c, a in zip(correctness_list, actuals) if a is not None]
    overall_accuracy = float(np.mean(valid_correctness) * 100) if valid_correctness else 0.0
    
    # Basic error statistics
    if valid_errors:
        mae = float(np.mean(valid_absolute_errors))
        mse = float(np.mean(valid_squared_errors))
        rmse = float(np.sqrt(mse))
        mape = float(np.mean(valid_percentage_errors))
        mean_error = float(np.mean(valid_errors))
        std_error = float(np.std(valid_errors))
        median_error = float(np.median(valid_errors))
        
        # R-squared calculation
        ss_res = np.sum((np.array(valid_predictions) - np.array(valid_actuals)) ** 2)
        ss_tot = np.sum((np.array(valid_actuals) - np.mean(valid_actuals)) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
        
        # Mean Absolute Percentage Error (MAPE) - alternative calculation
        valid_for_mape = [(p, a) for p, a in zip(valid_predictions, valid_actuals) if a != 0]
        if valid_for_mape:
            mape_alt = float(np.mean([abs(p - a) / a * 100 for p, a in valid_for_mape]))
        else:
            mape_alt = 0.0
        
        # Directional accuracy (sign of change prediction)
        directional_correct = 0
        for i in range(1, len(valid_actuals)):
            actual_change = valid_actuals[i] - valid_actuals[i-1]
            predicted_change = valid_predictions[i] - valid_actuals[i-1]
            if (actual_change >= 0 and predicted_change >= 0) or (actual_change < 0 and predicted_change < 0):
                directional_correct += 1
        directional_accuracy = (directional_correct / (len(valid_actuals) - 1)) * 100 if len(valid_actuals) > 1 else 0.0
        
        # Model-wise statistics
        model_stats = {}
        for model in ['low', 'mid', 'high']:
            model_indices = [i for i in valid_indices if model_used_list[i] == model]
            if model_indices:
                model_preds = [predictions[i] for i in model_indices]
                model_actuals = [actuals[i] for i in model_indices]
                model_mae = float(np.mean([abs(p - a) for p, a in zip(model_preds, model_actuals)]))
                model_count = len(model_indices)
                model_stats[model] = {
                    "count": model_count,
                    "mae": round(model_mae, 2),
                    "avg_prediction": round(np.mean(model_preds), 2),
                    "avg_actual": round(np.mean(model_actuals), 2)
                }
    else:
        mae = mse = rmse = mape = mean_error = std_error = median_error = r_squared = mape_alt = directional_accuracy = 0.0
        model_stats = {}
    
    # Percentiles of absolute errors
    if valid_absolute_errors:
        error_percentiles = {
            "25th": float(np.percentile(valid_absolute_errors, 25)),
            "50th": float(np.percentile(valid_absolute_errors, 50)),
            "75th": float(np.percentile(valid_absolute_errors, 75)),
            "90th": float(np.percentile(valid_absolute_errors, 90)),
            "95th": float(np.percentile(valid_absolute_errors, 95)),
            "max": float(np.max(valid_absolute_errors))
        }
    else:
        error_percentiles = {}

    # -------------------------------
    # 6) Prepare response data
    # -------------------------------
    simulation_data = []
    for i in range(len(pred_dates)):
        day_data = {
            "day": i + 1,
            "date": pred_dates[i].isoformat(),
            "predicted_amount": round(predictions[i], 2),
            "actual_amount": round(actuals[i], 2) if actuals[i] is not None else None,
            "model_used": model_used_list[i],
            "correct": correctness_list[i],
            "window_tail": window_tails[i]
        }
            
        simulation_data.append(day_data)

    response = {
        "metadata": {
            "today": today.isoformat(),
            "first_prediction_date": first_pred_date.isoformat(),
            "max_days": max_days,
            "accuracy_threshold": accuracy,
            "overestimate_ok": overestimate_ok_bool,
            "simulation_period": f"{first_pred_date.isoformat()} to {(first_pred_date + timedelta(days=max_days-1)).isoformat()}"
        },
        "summary_statistics": {
            "comparison_days": len(valid_actuals),
            "missing_days": len([a for a in actuals if a is None]),
            "overall_accuracy": round(overall_accuracy, 2),
            "correct_predictions": sum(valid_correctness),
            "incorrect_predictions": len(valid_correctness) - sum(valid_correctness)
        },
        "error_statistics": {
            "mean_absolute_error": round(mae, 2),
            "mean_squared_error": round(mse, 2),
            "root_mean_squared_error": round(rmse, 2),
            "mean_absolute_percentage_error": round(mape, 2),
            "mape_alternative": round(mape_alt, 2),
            "mean_error": round(mean_error, 2),
            "median_error": round(median_error, 2),
            "error_standard_deviation": round(std_error, 2),
            "r_squared": round(r_squared, 4),
            "directional_accuracy": round(directional_accuracy, 2),
            "error_percentiles": {k: round(v, 2) for k, v in error_percentiles.items()}
        },
        "model_performance": model_stats,
        "data_summary": {
            "actual_values": {
                "mean": round(float(np.mean(valid_actuals)), 2) if valid_actuals else None,
                "median": round(float(np.median(valid_actuals)), 2) if valid_actuals else None,
                "std": round(float(np.std(valid_actuals)), 2) if valid_actuals else None,
                "min": round(float(np.min(valid_actuals)), 2) if valid_actuals else None,
                "max": round(float(np.max(valid_actuals)), 2) if valid_actuals else None
            },
            "predicted_values": {
                "mean": round(float(np.mean(valid_predictions)), 2) if valid_predictions else None,
                "median": round(float(np.median(valid_predictions)), 2) if valid_predictions else None,
                "std": round(float(np.std(valid_predictions)), 2) if valid_predictions else None,
                "min": round(float(np.min(valid_predictions)), 2) if valid_predictions else None,
                "max": round(float(np.max(valid_predictions)), 2) if valid_predictions else None
            }
        },
        "simulation": simulation_data
    }

    return JSONResponse(response)

#--------------------------------------------------------------------------------


# ------------------------
# 6) API endpoints - Single Model
# ------------------------

@app.post("/simulateTomorrowSingle")
def simulateTomorrowSingle(
    currentDateTime: str = Query(..., description="Current date in YYYY-MM-DD format"),
    accuracy: float = Query(0.30, description="Accuracy threshold as decimal (default 0.30 = 30%)")
):
    """
    Predict tomorrow's expense based on current date and historical data (Single Model)
    """
    try:
        if single_model is None:
            return JSONResponse({
                "error": "Single model not available. Please ensure lstm_expense_model_single.h5 exists."
            }, status_code=400)
            
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
        
        # Prepare input for prediction
        X = seq_to_model_input(scaled_recent, recent_dates)
        
        # Make prediction using single model
        pred_scaled = float(single_model.predict(X, verbose=0)[0][0])
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
                "model_used": "single",
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
                "last_actual_amount": round(float(recent_data[-1]), 2),
                "days_used_for_prediction": WINDOW_SIZE
            },
            "accuracy_threshold_used": accuracy,
            "model_type": "single-model"
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

@app.post("/simulateTomorrowAutoSingle")
def simulateTomorrowAutoSingle(accuracy: float = Query(0.30, description="Accuracy threshold as decimal (default 0.30 = 30%)")):
    """
    Predict tomorrow's expense using today's date automatically (Single Model)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return simulateTomorrowSingle(today, accuracy)

@app.post("/simulateSingle")
def simulateSingle(
    max_days: int = Query(None, description="Limit how many days to simulate (default = 30)"),
    accuracy: float = Query(0.30, description="Accuracy threshold as decimal (default 0.30 = 30%)"),
    overestimate_ok: int = Query(0, description="If set to 1, considers overestimates as correct (default 0)")
):
    """
    Rolling future simulation starting from TOMORROW (PC date + 1). (Single Model)
    Includes detailed console logging for debugging.
    """

    if single_model is None:
        return JSONResponse({
            "error": "Single model not available. Please ensure lstm_expense_model_single.h5 exists."
        }, status_code=400)

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

    # Validate overestimate_ok parameter
    if overestimate_ok not in [0, 1]:
        return JSONResponse({
            "error": "overestimate_ok parameter must be 0 or 1"
        }, status_code=400)

    # Convert to boolean
    overestimate_ok_bool = bool(overestimate_ok)

    # -------------------------------
    # 1) Detect today + 1 (first prediction day)
    # -------------------------------
    today = datetime.now().date()
    first_pred_date = today + timedelta(days=1)

    if max_days is None or max_days <= 0:
        max_days = 30

    print("\n=== SIMULATION DEBUG START (Single Model) ===")
    print(f"PC Today: {today}")
    print(f"Prediction Start Day: {first_pred_date}")
    print(f"Simulating {max_days} days")
    print(f"Accuracy threshold: {accuracy*100}%")
    print(f"Overestimate considered correct: {overestimate_ok_bool}")
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

        # -------- Prediction --------
        X = seq_to_model_input(recent_scaled, recent_dates)
        pred_scaled = float(single_model.predict(X, verbose=0)[0][0])
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
    # 5) Correctness mask using the provided accuracy parameter and overestimate setting
    # -------------------------------
    correctness = []
    for p, a in zip(preds, actuals):
        if a is None:
            correctness.append(False)
        elif a == 0:
            correctness.append(True)
        else:
            if overestimate_ok_bool:
                # Consider correct if within accuracy threshold OR prediction is higher than actual
                error_ratio = abs(p - a) / a
                correctness.append(error_ratio <= accuracy or p >= a)
            else:
                # Original logic - only within accuracy threshold
                correctness.append(abs(p - a) / a <= accuracy)

    valid_mask = [c for c, a in zip(correctness, actuals) if a is not None]
    overall_accuracy = float(np.mean(valid_mask) * 100) if valid_mask else 0.0

    # -------------------------------
    # 6) Plot result - plot zeros for missing dates
    # -------------------------------
    # Replace None values with 0 for plotting
    actuals_clean = [0 if a is None else a for a in actuals]
    buf = plot_results(pred_dates, actuals_clean, preds, correctness, overall_accuracy, accuracy, overestimate_ok_bool, single_model=True)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

# ------------------------
# 7) Root
# ------------------------
@app.get("/")
def root():
    endpoints = {
        "multi_model_endpoints": {
            "simulate": "POST /simulate (optional query params max_days, accuracy, overestimate_ok) - returns PNG plot",
            "simulateFuture": "POST /simulateFuture (optional query params max_days, accuracy, overestimate_ok) - returns PNG plot", 
            "simulateTomorrow": "POST /simulateTomorrow?currentDateTime=YYYY-MM-DD&accuracy=0.30 - predict tomorrow",
            "simulateTomorrowAuto": "POST /simulateTomorrowAuto?accuracy=0.30 - predict tomorrow using today's date"
        },
        "single_model_endpoints": {
            "simulateSingle": "POST /simulateSingle (optional query params max_days, accuracy, overestimate_ok) - returns PNG plot",
            "simulateFutureSingle": "POST /simulateFutureSingle (optional query params max_days, accuracy, overestimate_ok) - returns PNG plot", 
            "simulateTomorrowSingle": "POST /simulateTomorrowSingle?currentDateTime=YYYY-MM-DD&accuracy=0.30 - predict tomorrow",
            "simulateTomorrowAutoSingle": "POST /simulateTomorrowAutoSingle?accuracy=0.30 - predict tomorrow using today's date"
        }
    }
    
    return {
        "message": "Rolling simulation ready with both multi-model and single-model support.",
        "endpoints": endpoints,
        "notes": "Simulation aggregates multiple transactions per day into daily totals.",
        "current_data_stats": {
            "total_days": len(amounts_daily),
            "date_range": {
                "start": dates_daily[0].strftime("%Y-%m-%d") if dates_daily else "No data",
                "end": dates_daily[-1].strftime("%Y-%m-%d") if dates_daily else "No data"
            },
            "window_size": WINDOW_SIZE,
            "accuracy_threshold": f"±{int(PCT_THRESHOLD*100)}%",
            "models_loaded": {
                "multi_models": list(models.keys()),
                "single_model": "loaded" if single_model is not None else "not loaded"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)