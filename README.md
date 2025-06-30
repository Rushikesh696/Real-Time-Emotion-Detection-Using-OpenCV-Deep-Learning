# Real-Time Emotion Detection Using OpenCV & Deep Learning

This project performs **real-time emotion recognition** using your webcam. It detects faces and classifies facial expressions into categories like *Happy, Sad, Angry, Surprise, Neutral*, etc., using a deep learning model and OpenCV.

---

## Features

- Real-time **face detection**
- Emotion classification on each detected face
- User-friendly interface via `app.py`
- Uses **CNN-based emotion detection model**
- Ready for extension with **voice feedback**, **streamlit integration**, or **gesture response**

---

## Technologies Used

- **Python**
- **OpenCV** – for face detection and video processing
- **TensorFlow/Keras or PyTorch** – for loading the trained emotion model
- **Pretrained model** (e.g., FER2013 dataset)
- Optional: `pyttsx3` for voice feedback

---

## Role of OpenCV and Deep Learning

### OpenCV
- Captures video feed from webcam
- Detects faces using `cv2.CascadeClassifier` or DNN face detector
- Draws rectangles and labels around detected faces

### Deep Learning Model
- Classifies cropped face images into **emotion categories**
- Trained on a dataset like FER-2013
- Outputs probabilities for each emotion label

