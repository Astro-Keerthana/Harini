from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

# Replace with your actual Firebase Realtime Database endpoint
FIREBASE_API_URL = "https://<your-firebase-project>.firebaseio.com/"

@app.route("/")
def home():
    return "Welcome to the Integrated Platform!"

@app.route("/api/data1")
def data1():
    try:
        response = requests.get(FIREBASE_API_URL + "data1.json")
        response.raise_for_status()  # Raise an error for bad responses
        return jsonify(response.json()), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/merged_data")
def merged_data():
    try:
        data1 = requests.get(FIREBASE_API_URL + "data1.json").json()
        
        
        merged = {
            "data1": data1
            
        }
        return jsonify(merged), 200
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
