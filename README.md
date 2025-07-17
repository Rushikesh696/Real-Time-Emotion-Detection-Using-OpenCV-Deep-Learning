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
**Haarcascade** is a face detection algorithm in OpenCV.

Loads a Haar cascade classifier for detecting faces in an image.

The XML file contains pre-trained data for recognizing human faces.

It uses Haar-like features (like edges and lines) to detect objects in an image.

Works by scanning the image at different scales and positions.

It uses a cascade of classifiers:

Simple classifiers quickly reject regions that don’t look like a face.

More complex classifiers focus only on likely face regions.

It’s fast and works in real-time on CPUs.

Great for detecting frontal faces.

**Limitations:** Struggles with tilted faces, poor lighting, or occlusion.



### OpenCV
- Captures video feed from webcam
- Detects faces using `cv2.CascadeClassifier` or DNN face detector
- Draws rectangles and labels around detected faces

### Deep Learning Model
- Classifies cropped face images into **emotion categories**
- Trained on a dataset like FER-2013
- Outputs probabilities for each emotion label

### Mini-Xception Model
Mini-Xception is a lightweight deep learning model for emotion recognition.

Xception stands for "Extreme Inception."

It’s a CNN architecture built using depthwise separable convolutions instead of regular convolutions.

The Mini version is a lightweight variation of Xception:

Designed for real-time emotion recognition on devices without GPUs.

Trained on the FER2013 dataset (Facial Expression Recognition dataset with 35,887 grayscale images, 48x48 pixels).

It’s a smaller version of the Xception CNN architecture (uses depthwise separable convolutions to make it faster and more efficient).

Takes a grayscale face image (64x64) as input.

Predicts one of 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

Trained on the FER-2013 dataset

Designed for real-time emotion detection.

Works well on devices without GPU because it’s lightweight and fast.

#### Architecture Overview

Input: Grayscale image (64x64x1).

Convolutional Layers: Extract features from the face (like eyes, mouth, etc.).

Depthwise Separable Convolutions: Reduces computation while keeping accuracy.

Fully Connected Layer: Maps features to emotions.

Softmax Output: Predicts probabilities for 7 emotions:

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral
