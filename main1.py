import cv2
from gaze_tracking import GazeTracking
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

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
