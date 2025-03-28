from flask import Flask, render_template, jsonify
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ThingSpeak Configuration
THING_SPEAK_CHANNEL_ID = "2855545"
THING_SPEAK_API_KEY = "YLFVDZN8HIM7850I"
THING_SPEAK_URL = f"https://api.thingspeak.com/channels/{THING_SPEAK_CHANNEL_ID}/feeds.json?api_key={THING_SPEAK_API_KEY}&results=100"

# Global data storage
historical_data = {
    'timestamps': [], 
    'voltage': [], 
    'current': [],
    'temperature': [], 
    'lux': [], 
    'power': []
}

# ML Models
models = {
    'linear_reg': None,
    'random_forest': None,
    'isolation_forest': None
}

scaler = StandardScaler()
feature_names = ['voltage', 'current', 'temperature', 'lux', 'hour']

def prepare_data():
    """Prepare DataFrame for ML models with proper feature names"""
    if not historical_data['timestamps']:
        return pd.DataFrame()
    
    df = pd.DataFrame(historical_data)
    try:
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df['hour'] = df['timestamps'].dt.hour
        # Ensure we only keep the columns we need with proper names
        df = df[['voltage', 'current', 'temperature', 'lux', 'hour', 'power']]
    except Exception as e:
        print(f"Error preparing data: {e}")
        return pd.DataFrame()
    return df

def train_models():
    """Train all ML models with proper feature naming"""
    df = prepare_data()
    if df.empty or len(df) < 10:
        return

    try:
        # Linear Regression
        X = np.array(range(len(df))).reshape(-1, 1)
        models['linear_reg'] = LinearRegression().fit(X, df['power'])

        # Random Forest Classifier
        if len(df) >= 30:
            X_rf = df[feature_names[:-1]]  # Exclude 'hour' for RF
            y_rf = pd.cut(df['power'], 
                         bins=[-1, 50, 100, float('inf')], 
                         labels=[0, 1, 2])
            models['random_forest'] = RandomForestClassifier(n_estimators=50).fit(X_rf, y_rf)

        # Isolation Forest
        if len(df) >= 20:
            X_iso = df[feature_names[:-2]]  # Only use voltage, current, temperature, lux
            X_scaled = scaler.fit_transform(X_iso)
            models['isolation_forest'] = IsolationForest(contamination=0.05).fit(X_scaled)
    except Exception as e:
        print(f"Error training models: {e}")

def get_predictions():
    """Generate predictions with proper feature naming"""
    predictions = {}
    df = prepare_data()
    if df.empty:
        return predictions

    try:
        # Linear Regression Prediction
        if models['linear_reg']:
            next_step = len(df)
            pred = models['linear_reg'].predict([[next_step]])[0]
            predictions['linear_pred'] = max(0, float(pred))

        # Efficiency Classification
        if models['random_forest'] and not df.empty:
            last_row = df.iloc[-1][feature_names[:-1]].values.reshape(1, -1)
            eff = models['random_forest'].predict(last_row)[0]
            predictions['efficiency'] = ['Low', 'Medium', 'High'][eff]

        # Anomaly Detection
        if models['isolation_forest'] and not df.empty:
            last_row = df.iloc[-1][feature_names[:-2]].values.reshape(1, -1)
            X_scaled = scaler.transform(last_row)
            is_anomaly = models['isolation_forest'].predict(X_scaled)[0] == -1
            predictions['anomaly'] = bool(is_anomaly)

    except Exception as e:
        print(f"Error generating predictions: {e}")

    return predictions

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    try:
        # Fetch data from ThingSpeak
        response = requests.get(THING_SPEAK_URL, timeout=5)
        response.raise_for_status()
        data = response.json()

        if 'feeds' not in data or not isinstance(data['feeds'], list):
            return jsonify({"error": "Invalid data format from ThingSpeak"}), 500

        # Clear previous data
        for key in historical_data:
            historical_data[key] = []

        # Process feeds
        for feed in data['feeds']:
            try:
                if all(field in feed for field in ['field1', 'field2', 'field3', 'field4']):
                    historical_data['timestamps'].append(feed['created_at'])
                    historical_data['voltage'].append(float(feed['field1']))
                    historical_data['current'].append(float(feed['field2']))
                    historical_data['temperature'].append(float(feed['field3']))
                    historical_data['lux'].append(float(feed['field4']))
                    historical_data['power'].append(float(feed['field1']) * float(feed['field2']))
            except (ValueError, KeyError) as e:
                print(f"Skipping invalid feed entry: {e}")
                continue

        if not historical_data['timestamps']:
            return jsonify({"error": "No valid data points found"}), 500

        # Train models
        train_models()

        # Prepare response
        latest = {
            "voltage": historical_data['voltage'][-1] if historical_data['voltage'] else 0,
            "current": historical_data['current'][-1] if historical_data['current'] else 0,
            "temperature": historical_data['temperature'][-1] if historical_data['temperature'] else 0,
            "lux": historical_data['lux'][-1] if historical_data['lux'] else 0,
            "power": historical_data['power'][-1] if historical_data['power'] else 0,
            "timestamp": historical_data['timestamps'][-1] if historical_data['timestamps'] else ""
        }

        # Add predictions
        predictions = get_predictions()
        response_data = {**latest, **predictions}

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"ThingSpeak request failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)