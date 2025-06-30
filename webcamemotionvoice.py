# webcam_emotion_voice.py

import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load Haarcascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# Load Mini-Xception emotion model
model = load_model('model/fer2013_mini_XCEPTION.102-0.66.hdf5')

# Emotion labels (ordered as per FER-2013 training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Track last spoken emotion to avoid repeating
last_emotion = None

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        # Resize to 64x64 as required by the model
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)  # Add channel dim for grayscale

        # Predict emotion
        preds = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(preds)]

        # Speak if emotion changes
        if label != last_emotion:
            print(f"Emotion: {label}")
            engine.say(f"You look {label.lower()}")
            engine.runAndWait()
            last_emotion = label

        # Draw face and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (36, 255, 12), 2)

    # Show the frame
    cv2.imshow("Emotion Detection (Press 'q' to exit)", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
