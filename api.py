from flask import Flask, jsonify, request
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
import json
import tempfile
import firebase_admin
from firebase_admin import credentials, storage
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()
AUTH_TOKEN = os.getenv('authentication')

# Initialize Firebase Admin SDK
cred = credentials.Certificate("hackfest-2025-7f1a4-firebase-adminsdk-fbsvc-a86f273b3e.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'hackfest-2025-7f1a4.firebasestorage.app'
})
bucket = storage.bucket()

app = Flask(__name__)

# Define paths for all models (removing general)
MODEL_PATHS = {
    'supply': 'models/prophet_model_supply.joblib',
    'demand': 'models/prophet_model_demand.joblib',
    'price': 'models/prophet_model_price.joblib'
}

# Add a model cache dictionary to store loaded models in memory
MODEL_CACHE = {}

FORECAST_CACHE_PATHS = {
    'supply': 'forecasts/latest_forecast_supply.json',
    'demand': 'forecasts/latest_forecast_demand.json',
    'price': 'forecasts/latest_forecast_price.json'
}

def load_model(model_type):
    """Load the trained Prophet model for specific type from Firebase or use cached version if available"""
    try:
        # Check if model is already in memory cache
        if model_type in MODEL_CACHE:
            print(f"Using cached {model_type} model from memory")
            return MODEL_CACHE[model_type]
            
        path = MODEL_PATHS.get(model_type)
        if not path:
            return None
            
        # Create a temporary file to store the downloaded model
        temp_model_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_name = temp_model_file.name
        temp_model_file.close()  # Close file handle immediately
        
        try:
            # Download the model from Firebase
            blob = bucket.blob(path)
            if not blob.exists():
                return None
                
            blob.download_to_filename(temp_file_name)
            
            # Load the model
            model = joblib.load(temp_file_name)
            
            # Store in cache for future use
            MODEL_CACHE[model_type] = model
            
            return model
        finally:
            # Clean up temporary file with improved error handling
            try:
                if os.path.exists(temp_file_name):
                    os.unlink(temp_file_name)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file_name}: {str(e)}")
                # Continue execution even if file deletion fails
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def save_model(model, model_type):
    """Save the trained model to Firebase"""
    try:
        path = MODEL_PATHS.get(model_type)
        if not path:
            return False
            
        # Save model to a temporary file first
        temp_model_file = tempfile.NamedTemporaryFile(delete=False)
        joblib.dump(model, temp_model_file.name)
        
        # Upload to Firebase
        blob = bucket.blob(path)
        blob.upload_from_filename(temp_model_file.name)
        
        # Clean up temporary file
        os.unlink(temp_model_file.name)
        
        # Update cache with the new model
        MODEL_CACHE[model_type] = model
        
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_cached_forecast(forecast_type):
    """Load the cached forecast from Firebase if available"""
    try:
        path = FORECAST_CACHE_PATHS.get(forecast_type)
        if not path:
            return None
            
        # Download the forecast JSON from Firebase
        blob = bucket.blob(path)
        if not blob.exists():
            return None
            
        # Download as string and parse JSON
        json_data = blob.download_as_string()
        return json.loads(json_data)
    except Exception as e:
        print(f"Error loading forecast: {str(e)}")
        return None

def save_forecast_cache(forecast_data, forecast_type):
    """Save forecast cache to Firebase"""
    try:
        path = FORECAST_CACHE_PATHS.get(forecast_type)
        if not path:
            return False
            
        # Convert to JSON string
        json_data = json.dumps(forecast_data)
        
        # Upload to Firebase
        blob = bucket.blob(path)
        blob.upload_from_string(json_data, content_type='application/json')
        
        return True
    except Exception as e:
        print(f"Error saving forecast: {str(e)}")
        return False

def process_json_file(json_data, data_type):
    """Process uploaded JSON file into a DataFrame ready for Prophet"""
    try:
        # If it's a string (file content), parse it
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
            
        # Extract data based on the expected JSON structure
        if "data" in data:
            # Format with nested data array like our JSON files
            items = data["data"]
            df = pd.DataFrame(items)
            # Convert to prophet format
            df['ds'] = pd.to_datetime(df['date'])
            df['y'] = df['value']
            return df[['ds', 'y']]
        elif isinstance(data, list):
            # Format with direct array of objects
            df = pd.DataFrame(data)
            # Check if keys exist
            if 'date' in df.columns and 'value' in df.columns:
                df['ds'] = pd.to_datetime(df['date'])
                df['y'] = df['value']
                return df[['ds', 'y']]
        
        raise ValueError("JSON data must contain 'data' array with 'date' and 'value' fields")
    
    except Exception as e:
        raise ValueError(f"Error processing JSON: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the API is running"""
    models_status = {}
    for model_type, path in MODEL_PATHS.items():
        # Check if model exists in Firebase
        blob = bucket.blob(path)
        models_status[model_type] = blob.exists()
        
    forecasts_status = {}
    for forecast_type, path in FORECAST_CACHE_PATHS.items():
        # Check if forecast exists in Firebase
        blob = bucket.blob(path)
        forecasts_status[forecast_type] = blob.exists()
    
    return jsonify({
        "status": "API is running", 
        "firebase_connected": True,
        "models_loaded": models_status,
        "cached_forecasts": forecasts_status
    })

@app.route('/update-forecast/<string:data_type>', methods=['POST'])
def update_forecast(data_type):
    """
    Update the forecast for a specific data type with uploaded data
    
    Expected multipart form data:
    - authentication: Your auth token
    - retrain: true/false  
    - data_file: JSON file with time series data (required)
    """
    try:
        # Check if data_type is valid
        if data_type not in MODEL_PATHS:
            return jsonify({
                "status": "error", 
                "message": f"Invalid data type. Must be one of: {list(MODEL_PATHS.keys())}"
            }), 400
        
        # Check authentication
        auth_token = request.form.get('authentication')
        if not auth_token or auth_token != AUTH_TOKEN:
            return jsonify({
                "status": "error",
                "message": "Authentication failed. Please provide a valid authentication token."
            }), 401
        
        # Check if retrain option is provided
        retrain = request.form.get('retrain', 'false').lower() == 'true'
        
        # Load the appropriate model
        model = load_model(data_type)
        
        if model is None and not retrain:
            return jsonify({
                "error": f"Model for {data_type} not found. Please train and export the model first."
            }), 404
        
        # Process data file (required now)
        if 'data_file' not in request.files:
            return jsonify({
                "error": "No data file provided. Please upload a JSON data file."
            }), 400
            
        file = request.files['data_file']
        json_data = json.load(file)
        input_data = process_json_file(json_data, data_type)
        
        if retrain or model is None:
            from prophet import Prophet
            
            # Create a new model with appropriate parameters
            new_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                interval_width=0.95
            )
            
            # Fit the new model with data
            new_model.fit(input_data)
            
            # Save the updated model to Firebase
            save_success = save_model(new_model, data_type)
            if not save_success:
                return jsonify({
                    "error": f"Failed to save model for {data_type} to Firebase."
                }), 500
                
            model = new_model
            retraining_message = f"Model for {data_type} retrained with provided data"
        else:
            retraining_message = f"Used existing {data_type} model (no retraining)"
        
        # Generate the forecast
        future = model.make_future_dataframe(periods=6, freq='ME')
        forecast = model.predict(future)
        
        # Extract forecast
        forecast_result = forecast.tail(6)
        
        # Create response data
        response = []
        for _, row in forecast_result.iterrows():
            response.append({
                "date": row['ds'].strftime('%Y-%m-%d'),
                "forecast": float(row['yhat']),
                "lower_bound": float(row['yhat_lower']),
                "upper_bound": float(row['yhat_upper'])
            })
        
        # Create forecast data
        forecast_data = {
            "forecast_period": f"Next 6 months from {datetime.now().strftime('%Y-%m-%d')}",
            "data_type": data_type,
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "retrained": retrain,
            "data": response
        }
        
        # Save the forecast cache to Firebase
        save_success = save_forecast_cache(forecast_data, data_type)
        if not save_success:
            return jsonify({
                "warning": f"Forecast generated successfully but failed to cache it in Firebase."
            })
        
        return jsonify({
            "status": "success",
            "message": f"Forecast updated successfully. {retraining_message}",
            "forecast": forecast_data
        })
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating forecast: {str(e)}"
        }), 500

@app.route('/forecast/<string:data_type>', methods=['GET'])
def get_forecast(data_type):
    """
    Get forecast for a specific data type
    Returns JSON with date and forecasted values
    """
    # Check if data_type is valid
    if data_type not in MODEL_PATHS:
        return jsonify({
            "status": "error", 
            "message": f"Invalid data type. Must be one of: {list(MODEL_PATHS.keys())}"
        }), 400
    
    # Try to load from Firebase cache first
    cached_forecast = load_cached_forecast(data_type)
    if cached_forecast:
        return jsonify(cached_forecast)
    
    # If no cache exists, generate a new forecast
    model = load_model(data_type)
    
    if model is None:
        return jsonify({
            "error": f"Model for {data_type} not found. Please train and export the model first."
        }), 404
    
    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=6, freq='ME')
    forecast = model.predict(future)
    
    # Extract forecast
    forecast_result = forecast.tail(6)
    
    # Create response data
    response = []
    for _, row in forecast_result.iterrows():
        response.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "forecast": float(row['yhat']),
            "lower_bound": float(row['yhat_lower']),
            "upper_bound": float(row['yhat_upper'])
        })
    
    # Create forecast data
    forecast_data = {
        "forecast_period": f"Next 6 months from {datetime.now().strftime('%Y-%m-%d')}",
        "data_type": data_type,
        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "data": response
    }
    
    # Save the forecast cache to Firebase
    save_forecast_cache(forecast_data, data_type)
    
    return jsonify(forecast_data)

@app.route('/forecast/custom/<string:data_type>', methods=['POST'])
def custom_forecast(data_type):
    """
    Get custom forecast for a specific data type
    
    Expected JSON input:
    {
        "periods": 6  # number of months to forecast (1-24)
    }
    """
    # Check if data_type is valid
    if data_type not in MODEL_PATHS:
        return jsonify({
            "status": "error", 
            "message": f"Invalid data type. Must be one of: {list(MODEL_PATHS.keys())}"
        }), 400
    
    # Get request data
    data = request.get_json()
    periods = data.get('periods', 6)
    
    # Load the appropriate model
    model = load_model(data_type)
    
    if model is None:
        return jsonify({
            "error": f"Model for {data_type} not found. Please train and export the model first."
        }), 404
    
    # Validate inputs
    if not isinstance(periods, int) or periods <= 0 or periods > 24:
        return jsonify({"error": "Periods must be a positive integer between 1 and 24"}), 400
    
    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=periods, freq='ME')
    forecast = model.predict(future)
    
    # Extract forecast
    forecast_result = forecast.tail(periods)
    
    # Create response data
    response = []
    for _, row in forecast_result.iterrows():
        response.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "forecast": float(row['yhat']),
            "lower_bound": float(row['yhat_lower']),
            "upper_bound": float(row['yhat_upper'])
        })
    
    return jsonify({
        "forecast_period": f"Next {periods} months from {datetime.now().strftime('%Y-%m-%d')}",
        "data_type": data_type,
        "data": response
    })

@app.route('/update-all-models', methods=['POST'])
def update_all_models():
    """
    Update all models with latest data
    
    Expected multipart form data:
    - authentication: Your auth token
    - retrain: true/false
    - supply_data: JSON file for supply (required)
    - demand_data: JSON file for demand (required) 
    - price_data: JSON file for price (required)
    """
    try:
        # Check authentication
        auth_token = request.form.get('authentication')
        if not auth_token or auth_token != AUTH_TOKEN:
            return jsonify({
                "status": "error",
                "message": "Authentication failed. Please provide a valid authentication token."
            }), 401
        
        retrain = request.form.get('retrain', 'false').lower() == 'true'
        results = {}
        
        # Check for required files
        required_files = ['supply_data', 'demand_data', 'price_data']
        for file_key in required_files:
            if file_key not in request.files:
                return jsonify({
                    "status": "error", 
                    "message": f"Missing required file: {file_key}"
                }), 400
        
        # Process each model type
        for data_type in ['supply', 'demand', 'price']:
            try:
                # Load model if not retraining
                model = None
                if not retrain:
                    model = load_model(data_type)
                    
                    if model is None:
                        results[data_type] = {"status": "error", "message": f"Model not found for {data_type}"}
                        continue
                
                # Process uploaded data
                file_key = f"{data_type}_data"
                file = request.files[file_key]
                json_data = json.load(file)
                input_data = process_json_file(json_data, data_type)
                
                if retrain or model is None:
                    from prophet import Prophet
                    
                    # Create and train new model
                    new_model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        seasonality_mode='additive',
                        interval_width=0.95
                    )
                    
                    new_model.fit(input_data)
                    save_success = save_model(new_model, data_type)
                    
                    if not save_success:
                        results[data_type] = {"status": "error", "message": f"Failed to save model to Firebase"}
                        continue
                        
                    model = new_model
                
                # Generate forecast
                future = model.make_future_dataframe(periods=6, freq='ME')
                forecast = model.predict(future)
                forecast_result = forecast.tail(6)
                
                # Create response
                response = []
                for _, row in forecast_result.iterrows():
                    response.append({
                        "date": row['ds'].strftime('%Y-%m-%d'),
                        "forecast": float(row['yhat']),
                        "lower_bound": float(row['yhat_lower']),
                        "upper_bound": float(row['yhat_upper'])
                    })
                
                forecast_data = {
                    "forecast_period": f"Next 6 months from {datetime.now().strftime('%Y-%m-%d')}",
                    "data_type": data_type,
                    "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "retrained": retrain,
                    "data": response
                }
                
                # Save forecast cache to Firebase
                save_success = save_forecast_cache(forecast_data, data_type)
                
                results[data_type] = {
                    "status": "success", 
                    "message": f"{'Retrained and updated' if retrain else 'Updated'} forecast for {data_type}",
                    "cache_saved": save_success
                }
                
            except Exception as e:
                results[data_type] = {"status": "error", "message": str(e)}
        
        return jsonify({
            "status": "complete",
            "results": results
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error updating models: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    print("API is running on http://localhost:5000")
