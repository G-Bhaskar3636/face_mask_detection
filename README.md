# 😷 Face Mask Detection using VGG16

A deep learning project that detects whether a person is wearing a face mask using **VGG16 Transfer Learning** and a **Streamlit** web application.

## 🚀 Project Overview

This project uses a pre-trained VGG16 model to classify faces into two categories:

* 😷 With Mask
* 🚫 Without Mask

The trained model is integrated into a Streamlit application where users can use their camera to capture an image and get a mask prediction.

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* VGG16
* OpenCV
* NumPy
* Streamlit
* Pillow

## 📂 Project Structure

```text
mask_detection/
│
├── main.py
├── mask_detector.keras
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔄 Workflow

```text
Training Dataset
       ↓
Image Preprocessing
       ↓
Train/Test Split
       ↓
VGG16 Transfer Learning
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Save Trained Model
       ↓
Streamlit Application
       ↓
Camera Input
       ↓
Face Detection
       ↓
Mask Prediction
```

## 🏗️ Model

The project uses **VGG16**, a pre-trained convolutional neural network.

The original final classification layer is replaced with:

```python
Dense(1, activation='sigmoid')
```

Since this is a binary classification problem:

```text
0 → With Mask
1 → Without Mask
```

The VGG16 feature-extraction layers are frozen during training.

## 📊 Model Performance

The model achieved approximately:

```text
Test Accuracy: 98.35%
```

### Confusion Matrix

```text
[[429   4]
 [ 11 464]]
```

This means:

* 429 `with_mask` images were correctly classified.
* 464 `without_mask` images were correctly classified.
* 4 `with_mask` images were incorrectly classified.
* 11 `without_mask` images were incorrectly classified.

### Classification Report

| Class        | Precision | Recall | F1-Score |
| ------------ | --------: | -----: | -------: |
| With Mask    |      0.97 |   0.99 |     0.98 |
| Without Mask |      0.99 |   0.98 |     0.98 |

## 💻 Run the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Go to the project folder

```bash
cd mask_detection
```


### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run Streamlit

```bash
streamlit run main.py
```

The application will open in your browser.

## 📷 How the Application Works

1. Open the Streamlit application.
2. Allow camera access.
3. Take a picture.
4. OpenCV detects the face.
5. The detected face is resized to `224 × 224`.
6. VGG16 preprocessing is applied.
7. The trained model predicts the class.
8. The application displays:

```text
With Mask
```

or

```text
Without Mask
```

## 📦 Model File

The trained model is saved as:

```text
mask_detector.keras
```

The Streamlit application loads this model instead of training the model again.

```python
model = load_model("mask_detector.keras")
```

## ⚠️ Limitations

* The current application is designed for basic mask detection.
* Haar Cascade may not detect every face correctly.
* Poor lighting can affect face detection.
* Multiple faces may produce different prediction results.
* The model's high accuracy depends on the quality and similarity of the training dataset.

## 🔮 Future Improvements

* Real-time continuous webcam detection.
* Better face detection using modern detectors.
* Improve performance with data augmentation.
* Add confidence scores.
* Support multiple faces more effectively.
* Add a more interactive Streamlit interface.
* Deploy the application online.

## 👨‍💻 Author

**Bhaskar**

This project was created as a deep learning and computer vision project using VGG16 transfer learning and Streamlit.
