from flask import Flask, jsonify, request
import pandas as pd
from datetime import datetime
import joblib
import os
import json
import tempfile
import firebase_admin
from firebase_admin import credentials, storage
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

# --- ENV & FIREBASE INIT ---
load_dotenv()
AUTH_TOKEN = os.getenv('authentication')

cred = credentials.Certificate("hackfest-2025-7f1a4-firebase-adminsdk-fbsvc-a86f273b3e.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'hackfest-2025-7f1a4.firebasestorage.app'
})
bucket = storage.bucket()

app = Flask(__name__)

MODEL_PATHS = {
    'supply': 'models/prophet_model_supply.joblib',
    'demand': 'models/prophet_model_demand.joblib',
    'price': 'models/prophet_model_price.joblib'
}
FORECAST_CACHE_PATHS = {
    'supply': 'forecasts/latest_forecast_supply.json',
    'demand': 'forecasts/latest_forecast_demand.json',
    'price': 'forecasts/latest_forecast_price.json'
}
MODEL_CACHE: Dict[str, Any] = {}

# --- UTILS ---

def get_blob(path: str):
    return bucket.blob(path)

def safe_json_load(file):
    try:
        return json.load(file)
    except Exception:
        file.seek(0)
        return json.loads(file.read().decode())

def process_json_file(json_data: Any, data_type: str) -> pd.DataFrame:
    try:
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        if "data" in data:
            items = data["data"]
            df = pd.DataFrame(items)
            df['ds'] = pd.to_datetime(df['date'])
            df['y'] = df['value']
            return df[['ds', 'y']]
        elif isinstance(data, list):
            df = pd.DataFrame(data)
            if 'date' in df.columns and 'value' in df.columns:
                df['ds'] = pd.to_datetime(df['date'])
                df['y'] = df['value']
                return df[['ds', 'y']]
        raise ValueError("JSON data must contain 'data' array with 'date' and 'value' fields")
    except Exception as e:
        raise ValueError(f"Error processing JSON: {str(e)}")

def create_future_dataframe(model, periods: int = 6) -> pd.DataFrame:
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

# --- MODEL/FORECAST HELPERS ---

def load_model(model_type: str) -> Optional[Any]:
    try:
        if model_type in MODEL_CACHE:
            print(f"Using cached {model_type} model from memory")
            return MODEL_CACHE[model_type]
        path = MODEL_PATHS.get(model_type)
        if not path:
            return None
        temp_model_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_name = temp_model_file.name
        temp_model_file.close()
        try:
            blob = bucket.blob(path)
            if not blob.exists():
                return None
            blob.download_to_filename(temp_file_name)
            model = joblib.load(temp_file_name)
            MODEL_CACHE[model_type] = model
            return model
        finally:
            try:
                if os.path.exists(temp_file_name):
                    os.unlink(temp_file_name)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file_name}: {str(e)}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def save_model(model: Any, model_type: str) -> bool:
    try:
        path = MODEL_PATHS.get(model_type)
        if not path:
            return False
        temp_model_file = tempfile.NamedTemporaryFile(delete=False)
        joblib.dump(model, temp_model_file.name)
        blob = bucket.blob(path)
        blob.upload_from_filename(temp_model_file.name)
        os.unlink(temp_model_file.name)
        MODEL_CACHE[model_type] = model
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_cached_forecast(forecast_type: str) -> Optional[dict]:
    try:
        path = FORECAST_CACHE_PATHS.get(forecast_type)
        if not path:
            return None
        blob = bucket.blob(path)
        if not blob.exists():
            return None
        json_data = blob.download_as_string()
        return json.loads(json_data)
    except Exception as e:
        print(f"Error loading forecast: {str(e)}")
        return None

def save_forecast_cache(forecast_data: dict, forecast_type: str) -> bool:
    try:
        path = FORECAST_CACHE_PATHS.get(forecast_type)
        if not path:
            return False
        json_data = json.dumps(forecast_data)
        blob = bucket.blob(path)
        blob.upload_from_string(json_data, content_type='application/json')
        return True
    except Exception as e:
        print(f"Error saving forecast: {str(e)}")
        return False

def build_forecast_response(forecast: pd.DataFrame) -> List[dict]:
    return [
        {
            "date": row['ds'].strftime('%Y-%m-%d'),
            "forecast": float(row['yhat']),
            "lower_bound": float(row['yhat_lower']),
            "upper_bound": float(row['yhat_upper'])
        }
        for _, row in forecast.iterrows()
    ]

# --- ROUTES ---

@app.route('/health', methods=['GET'])
def health_check():
    models_status = {k: get_blob(v).exists() for k, v in MODEL_PATHS.items()}
    forecasts_status = {k: get_blob(v).exists() for k, v in FORECAST_CACHE_PATHS.items()}
    return jsonify({
        "status": "API is running",
        "firebase_connected": True,
        "models_loaded": models_status,
        "cached_forecasts": forecasts_status
    })

@app.route('/update-forecast/<string:data_type>', methods=['POST'])
def update_forecast(data_type):
    if data_type not in MODEL_PATHS:
        return jsonify({"status": "error", "message": f"Invalid data type. Must be one of: {list(MODEL_PATHS.keys())}"}), 400
    if request.form.get('authentication') != AUTH_TOKEN:
        return jsonify({"status": "error", "message": "Authentication failed. Please provide a valid authentication token."}), 401
    retrain = request.form.get('retrain', 'false').lower() == 'true'
    model = load_model(data_type)
    if model is None and not retrain:
        return jsonify({"error": f"Model for {data_type} not found. Please train and export the model first."}), 404
    if 'data_file' not in request.files:
        return jsonify({"error": "No data file provided. Please upload a JSON data file."}), 400
    file = request.files['data_file']
    json_data = safe_json_load(file)
    input_data = process_json_file(json_data, data_type)
    if retrain or model is None:
        from prophet import Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            interval_width=0.95
        )
        model.fit(input_data)
        if not save_model(model, data_type):
            return jsonify({"error": f"Failed to save model for {data_type} to Firebase."}), 500
        retraining_message = f"Model for {data_type} retrained with provided data"
    else:
        retraining_message = f"Used existing {data_type} model (no retraining)"
    future = create_future_dataframe(model, periods=6)
    forecast = model.predict(future)
    response = build_forecast_response(forecast)
    forecast_data = {
        "forecast_period": f"Next 6 months from {datetime.now().strftime('%Y-%m-%d')}",
        "data_type": data_type,
        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "retrained": retrain,
        "data": response
    }
    save_forecast_cache(forecast_data, data_type)
    return jsonify({
        "status": "success",
        "message": f"Forecast updated successfully. {retraining_message}",
        "forecast": forecast_data
    })

@app.route('/forecast/<string:data_type>', methods=['GET'])
def get_forecast(data_type):
    if data_type not in MODEL_PATHS:
        return jsonify({"status": "error", "message": f"Invalid data type. Must be one of: {list(MODEL_PATHS.keys())}"}), 400
    cached_forecast = load_cached_forecast(data_type)
    if cached_forecast:
        return jsonify(cached_forecast)
    model = load_model(data_type)
    if model is None:
        return jsonify({"error": f"Model for {data_type} not found. Please train and export the model first."}), 404
    future = create_future_dataframe(model, periods=6)
    forecast = model.predict(future)
    response = build_forecast_response(forecast)
    forecast_data = {
        "forecast_period": f"Next 6 months from {datetime.now().strftime('%Y-%m-%d')}",
        "data_type": data_type,
        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "data": response
    }
    save_forecast_cache(forecast_data, data_type)
    return jsonify(forecast_data)

@app.route('/forecast/custom/<string:data_type>', methods=['POST'])
def custom_forecast(data_type):
    if data_type not in MODEL_PATHS:
        return jsonify({"status": "error", "message": f"Invalid data type. Must be one of: {list(MODEL_PATHS.keys())}"}), 400
    data = request.get_json()
    periods = data.get('periods', 6)
    model = load_model(data_type)
    if model is None:
        return jsonify({"error": f"Model for {data_type} not found. Please train and export the model first."}), 404
    if not isinstance(periods, int) or periods <= 0 or periods > 24:
        return jsonify({"error": "Periods must be a positive integer between 1 and 24"}), 400
    future = create_future_dataframe(model, periods=periods)
    forecast = model.predict(future)
    response = build_forecast_response(forecast)
    return jsonify({
        "forecast_period": f"Next {periods} months from {datetime.now().strftime('%Y-%m-%d')}",
        "data_type": data_type,
        "data": response
    })

@app.route('/update-all-models', methods=['POST'])
def update_all_models():
    if request.form.get('authentication') != AUTH_TOKEN:
        return jsonify({"status": "error", "message": "Authentication failed. Please provide a valid authentication token."}), 401
    retrain = request.form.get('retrain', 'false').lower() == 'true'
    required_files = ['supply_data', 'demand_data', 'price_data']
    for file_key in required_files:
        if file_key not in request.files:
            return jsonify({"status": "error", "message": f"Missing required file: {file_key}"}), 400
    results = {}
    for data_type in ['supply', 'demand', 'price']:
        try:
            model = None
            if not retrain:
                model = load_model(data_type)
                if model is None:
                    results[data_type] = {"status": "error", "message": f"Model not found for {data_type}"}
                    continue
            file_key = f"{data_type}_data"
            file = request.files[file_key]
            json_data = safe_json_load(file)
            input_data = process_json_file(json_data, data_type)
            if retrain or model is None:
                from prophet import Prophet
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    interval_width=0.95
                )
                model.fit(input_data)
                if not save_model(model, data_type):
                    results[data_type] = {"status": "error", "message": f"Failed to save model to Firebase"}
                    continue
            future = create_future_dataframe(model, periods=6)
            forecast = model.predict(future)
            response = build_forecast_response(forecast)
            forecast_data = {
                "forecast_period": f"Next 6 months from {datetime.now().strftime('%Y-%m-%d')}",
                "data_type": data_type,
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "retrained": retrain,
                "data": response
            }
            save_forecast_cache(forecast_data, data_type)
            results[data_type] = {
                "status": "success",
                "message": f"{'Retrained and updated' if retrain else 'Updated'} forecast for {data_type}",
                "cache_saved": True
            }
        except Exception as e:
            results[data_type] = {"status": "error", "message": str(e)}
    return jsonify({"status": "complete", "results": results})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
