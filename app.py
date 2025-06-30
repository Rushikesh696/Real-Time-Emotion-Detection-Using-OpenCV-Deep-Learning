import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Emotion labels
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Load the pre-trained Mini-Xception model
model = load_model("model/fer2013_mini_XCEPTION.102-0.66.hdf5")

# Load Haarcascade face detector
face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

# Load your test image
image_path = "test_images/IMG-20250425-WA0002.jpg"  
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Loop over all detected faces
for (x, y, w, h) in faces:
    roi_gray = gray[y:y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (64, 64))
    roi_gray = roi_gray.astype("float") / 255.0
    roi_gray = np.expand_dims(roi_gray, axis=0)
    roi_gray = np.expand_dims(roi_gray, axis=-1)

    # Predict emotion
    prediction = model.predict(roi_gray)
    label = emotion_labels[int(np.argmax(prediction))]

    # Draw bounding box and label
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Show the image
cv2.imshow("Emotion Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
