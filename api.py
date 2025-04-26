from flask import Flask, jsonify, request
import pandas as pd
from datetime import datetime
import joblib
import os
import json
import tempfile
import firebase_admin
from firebase_admin import credentials, storage, firestore
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# --- ENV & FIREBASE INIT ---
load_dotenv()
AUTH_TOKEN = os.getenv('authentication')

cred = credentials.Certificate("hackfest-2025-7f1a4-firebase-adminsdk-fbsvc-a86f273b3e.json")
firebase_admin.initialize_app(
    cred,
    {'storageBucket': 'hackfest-2025-7f1a4.firebasestorage.app'}
)
bucket = storage.bucket()

# --- FIRESTORE INIT ---
db = firestore.client()

app = Flask(__name__)

# --- HEALTH CHECK ENDPOINT FOR CLOUD RUN ---
@app.route('/healthz', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

# --- UTILS ---

def safe_json_load(file) -> Any:
    """Safely load JSON from a file-like object."""
    try:
        return json.load(file)
    except Exception:
        file.seek(0)
        return json.loads(file.read().decode())

def process_json_file(json_data: Any, data_type: str) -> pd.DataFrame:
    """
    Convert uploaded JSON data to a DataFrame suitable for Prophet.
    Expects either a dict with 'data' key or a list of dicts.
    """
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        if "data" in data:
            items = data["data"]
            df = pd.DataFrame(items)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("JSON data must contain 'data' array with 'date' and 'value' fields")
        if 'date' not in df.columns or 'value' not in df.columns:
            raise ValueError("Each item must have 'date' and 'value' fields")
        df['ds'] = pd.to_datetime(df['date'])
        df['y'] = df['value']
        return df[['ds', 'y']]
    except Exception as e:
        raise ValueError(f"Error processing JSON: {str(e)}")

def create_future_dataframe(model, periods: int = 6) -> pd.DataFrame:
    """
    Create a DataFrame of future dates (monthly, starting next month) for Prophet prediction.
    """
    current_date = datetime.now()
    if current_date.month == 12:
        next_month = datetime(current_date.year + 1, 1, 1)
    else:
        next_month = datetime(current_date.year, current_date.month + 1, 1)
    future_dates = []
    for i in range(periods):
        month = next_month.month + i
        year = next_month.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        future_dates.append(pd.Timestamp(year=year, month=month, day=28))
    return pd.DataFrame({'ds': future_dates})

def build_forecast_response(forecast: pd.DataFrame) -> List[dict]:
    """
    Convert Prophet forecast DataFrame to a list of dicts for API response.
    """
    return [
        {
            "date": row['ds'].strftime('%Y-%m-%d'),
            "forecast": float(row['yhat']),
            "lower_bound": float(row['yhat_lower']),
            "upper_bound": float(row['yhat_upper'])
        }
        for _, row in forecast.iterrows()
    ]

def load_model_from_firebase(model_path: str):
    """
    Download model from Firebase Storage and load with joblib.
    """
    try:
        blob = bucket.blob(model_path)
        if not blob.exists():
            return None
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            blob.download_to_filename(temp_file.name)
            model = joblib.load(temp_file.name)
        os.unlink(temp_file.name)
        return model
    except Exception as e:
        print(f"Error loading model from Firebase: {str(e)}")
        return None

# --- MAIN ENDPOINT ---

@app.route('/upload-product-forecast', methods=['POST'])
def upload_product_forecast():
    """
    Expects multipart/form-data:
      - authentication
      - productId
      - productName
      - supply_data: JSON file
      - demand_data: JSON file
      - price_data: JSON file
      - bulk_price_data: JSON file
    """
    # --- Auth & Input Validation ---
    if request.form.get('authentication') != AUTH_TOKEN:
        return jsonify({"status": "error", "message": "Authentication failed."}), 401

    product_id = request.form.get('productId')
    product_name = request.form.get('productName')
    if not product_id or not product_name:
        return jsonify({"status": "error", "message": "Missing productId or productName."}), 400

    required_files = ['supply_data', 'demand_data', 'price_data', 'bulk_price_data']
    for file_key in required_files:
        if file_key not in request.files:
            return jsonify({"status": "error", "message": f"Missing required file: {file_key}"}), 400

    # --- Model Training, Saving, and Forecast Processing ---
    forecasts = {}
    model_map = {
        'supply': 'prophet_model_supply.joblib',
        'demand': 'prophet_model_demand.joblib',
        'normalPrice': 'prophet_model_price.joblib',
        'bulkPrice': 'prophet_model_bulk_price.joblib'
    }
    data_file_map = {
        'supply': 'supply_data',
        'demand': 'demand_data',
        'normalPrice': 'price_data',
        'bulkPrice': 'bulk_price_data'
    }

    product_folder = f"models/{product_id}/"

    for key in ['supply', 'demand', 'normalPrice', 'bulkPrice']:
        try:
            file = request.files[data_file_map[key]]
            json_data = safe_json_load(file)
            input_data = process_json_file(json_data, key if key in ['supply', 'demand'] else 'price')

            # Build and fit model with optimized parameters
            from prophet import Prophet
            if key == 'supply':
                model = Prophet(
                    yearly_seasonality=10,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    interval_width=0.95,
                    mcmc_samples=0,
                    uncertainty_samples=100
                )
            elif key == 'demand':
                model = Prophet(
                    yearly_seasonality=10,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    mcmc_samples=0,
                    uncertainty_samples=100
                )
            else:  # normalPrice or bulkPrice
                model = Prophet(
                    yearly_seasonality=10,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.01,
                    seasonality_mode='multiplicative',
                    mcmc_samples=0,
                    uncertainty_samples=100
                )
            model.fit(input_data)

            # Save model to a temp file and upload to Firebase Storage in product-specific folder
            model_filename = model_map[key]
            model_path = product_folder + model_filename
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                joblib.dump(model, temp_file.name)
                blob = bucket.blob(model_path)
                blob.upload_from_filename(temp_file.name)
            os.unlink(temp_file.name)

            # Predict 6 months ahead
            future = create_future_dataframe(model, periods=6)
            forecast = model.predict(future)
            forecasts[key] = build_forecast_response(forecast)
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Error processing {key}: {str(e)}"
            }), 500

    # --- Fair Price Calculation for this month ---
    try:
        # Get the latest value from each input (current month)
        def get_latest_value(json_data):
            data = json.loads(json_data) if isinstance(json_data, str) else json_data
            if "data" in data:
                items = data["data"]
            elif isinstance(data, list):
                items = data
            else:
                return None
            if not items:
                return None
            return items[-1]["value"]

        supply_val = get_latest_value(safe_json_load(request.files['supply_data']))
        demand_val = get_latest_value(safe_json_load(request.files['demand_data']))
        price_val = get_latest_value(safe_json_load(request.files['price_data']))
        bulk_price_val = get_latest_value(safe_json_load(request.files['bulk_price_data']))

        alpha = 0.5
        fair_price = None
        bulk_fair_price = None
        if supply_val is not None and demand_val is not None and price_val is not None and supply_val != 0:
            fair_price = price_val + (alpha * ((demand_val - supply_val) / supply_val) * price_val)
        if supply_val is not None and demand_val is not None and bulk_price_val is not None and supply_val != 0:
            bulk_fair_price = bulk_price_val + (alpha * ((demand_val - supply_val) / supply_val) * bulk_price_val)
    except Exception as e:
        fair_price = None
        bulk_fair_price = None

    # --- Firestore Upload ---
    doc_data = {
        "productId": product_id,
        "productName": product_name,
        "supply": forecasts['supply'],
        "demand": forecasts['demand'],
        "normalPrice": forecasts['normalPrice'],
        "bulkPrice": forecasts['bulkPrice'],
        "thisMonthFairPrice": fair_price,
        "thisMonthBulkFairPrice": bulk_fair_price,
        "createdAt": datetime.now()
    }
    try:
        db.collection('product').add(doc_data)
        return jsonify({
            "status": "success",
            "message": "Forecast and models uploaded to Firestore and Firebase Storage.",
            "data": doc_data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Firestore error: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Use PORT env variable, default to 8080 for Cloud Run
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
