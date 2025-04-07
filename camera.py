import os
import gdown
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        model_path = "static/model.h5"
        json_path = "static/model.json"

        # Download model.h5 from Google Drive if it doesn't exist
        if not os.path.exists(model_path):
            print("Downloading model.h5 from Google Drive...")
            url = "https://drive.google.com/uc?id=1MPql4BPEmMBw9y0XJP66kk8SFxaTxG7j"
            gdown.download(url, model_path, quiet=False)

        # Load model architecture and weights
        self.model = model_from_json(open(json_path, "r").read())
        self.model.load_weights(model_path)

        # Load Haar Cascade
        self.face_cascade = cv2.CascadeClassifier("static/haarcascade_frontalface_default.xml")

        # Define emotions
        self.emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            prediction = self.model.predict(img_pixels)
            max_index = int(np.argmax(prediction))
            predicted_emotion = self.emotions[max_index]

            cv2.putText(frame, predicted_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
