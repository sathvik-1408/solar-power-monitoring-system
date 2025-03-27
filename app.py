from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)

# ThingSpeak API Configuration
THING_SPEAK_CHANNEL_ID = "2855545"  # Replace with your actual ThingSpeak channel ID
THING_SPEAK_API_KEY = "YLFVDZN8HIM7850I"   # Replace with your ThingSpeak Read API Key
THING_SPEAK_URL = f"https://api.thingspeak.com/channels/{THING_SPEAK_CHANNEL_ID}/feeds/last.json?api_key={THING_SPEAK_API_KEY}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    try:
        response = requests.get(THING_SPEAK_URL)
        data = response.json()

        if "field1" in data and "field2" in data and "field3" in data and "field4" in data:
            result = {
                "field1": float(data["field1"]),  # Voltage
                "field2": float(data["field2"]),  # Current
                "field3": float(data["field3"]),  # Temperature
                "field4": float(data["field4"])   # Light Intensity
            }
            return jsonify(result)
        return jsonify({"error": "Invalid data from ThingSpeak"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
