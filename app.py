from flask import Flask, render_template, request, jsonify, Response
import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import cv2
from gaze_tracking import GazeTracking
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI use

app = Flask(__name__)

# Parameters for the stress prediction model
solver = 'liblinear'
random_state = 1
output_file = 'final_model.bin'

# Most important features for the model
most_importan_features = ['limb_movement', 'sleeping_hours', 'snoring_rate', 'eye_movement', 'body_temperature']

# Stress level mapping
stress_level_mapping = {
    0: "No Stress",
    1: "Low Stress",
    2: "High Stress"
}

# Function to load the model from a file
def load_model(file_path):
    with open(file_path, 'rb') as f_in:
        return pickle.load(f_in)

# Function to predict stress level
def predict(user_data, dv, model):
    # Transform input data using the DictVectorizer
    X = dv.transform([user_data])
    y_pred = model.predict(X)
    return y_pred[0]

# Main function to load model and predict
def predict_stress_level(form_data):
    print("Loading the model...")
    dv, model = load_model(output_file)

    print("Model loaded successfully!")
    
    # Transform form data to match the input structure expected by the model
    user_data = {feature: float(form_data[feature]) for feature in most_importan_features}

    # Make prediction
    predicted_stress = predict(user_data, dv, model)
    stress_level_description = stress_level_mapping.get(predicted_stress, "Unknown Stress Level")
    
    return stress_level_description

# GazeTracking class
class GazeTrackerApp:
    def __init__(self):
        self.gaze = GazeTracking()
        self.webcam = None
        self.blink_count = 0
        self.right_count = 0
        self.left_count = 0
        self.center_count = 0

    def start_camera(self):
        if self.webcam is None:
            self.webcam = cv2.VideoCapture(0)

    def stop_camera(self):
        if self.webcam:
            self.webcam.release()
            self.webcam = None

    def get_frame(self):
        if not self.webcam:
            return None
        ret, frame = self.webcam.read()
        if not ret or frame is None:
            return None
        self.gaze.refresh(frame)
        frame = self.gaze.annotated_frame()

        # Update counts
        if self.gaze.is_blinking():
            self.blink_count += 1
        elif self.gaze.is_right():
            self.right_count += 1
        elif self.gaze.is_left():
            self.left_count += 1
        elif self.gaze.is_center():
            self.center_count += 1

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def generate_chart(self):
        # Create a bar chart showing total count for each label
        labels = ["Blinking", "Looking Right", "Looking Left", "Looking Center"]
        counts = [self.blink_count, self.right_count, self.left_count, self.center_count]

        # Create the bar chart
        plt.bar(labels, counts, color=['red', 'green', 'blue', 'orange'])

        # Add count labels on top of each bar
        for i, count in enumerate(counts):
            plt.text(i, count + 1, str(count), ha='center', va='bottom', fontsize=12)

        plt.xlabel('Labels')
        plt.ylabel('Count')

        # Save the chart to a BytesIO object
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()

        # Encode the image to base64 string
        img.seek(0)
        img_base64 = base64.b64encode(img.read()).decode('utf-8')

        # Determine focus result
        result = "User is focusing on the screen" if self.center_count == max(counts) else "User is not focusing on the screen"

        return img_base64, result

gaze_tracker = GazeTrackerApp()

# Route for the home page (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route for the Predict Stress Level page (predict_stress.html)
@app.route('/predict_stress', methods=['GET', 'POST'])
def predict_stress_view():
    if request.method == 'POST':
        # Get form data from POST request
        form_data = request.form
        
        # Call the predict_stress function
        prediction = predict_stress_level(form_data)
        
        # Return the template with the prediction
        return render_template('predict_stress.html', prediction=prediction)
    
    # Render the form if it's a GET request
    return render_template('predict_stress.html')

# Route for the Gaze Tracking page (eye_trac.html)
@app.route('/gaze_tracking')
def gaze_tracking():
    return render_template('eye_trac.html')

# Start the camera for gaze tracking
@app.route('/start_camera')
def start_camera():
    gaze_tracker.start_camera()
    return jsonify({"status": "Camera started"})

# Stop the camera and return analysis result and chart
@app.route('/stop_camera')
def stop_camera():
    gaze_tracker.stop_camera()
    img_base64, analysis_result = gaze_tracker.generate_chart()
    return jsonify({
        "result": analysis_result,
        "chart": img_base64
    })

# Video feed route for displaying the camera feed
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = gaze_tracker.get_frame()
            if frame is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
