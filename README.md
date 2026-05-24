# 🛡️ Multimodal Fake News Detection System

### Fake News + Fake Image + Fake Audio + Fake Video Detection using Machine Learning and Deep Learning

---

## 📌 Project Overview

The **Multimodal Fake News Detection System** is an advanced Artificial Intelligence project designed to identify and classify different types of fake or manipulated digital content. In today's digital era, misinformation spreads rapidly through news articles, manipulated images, AI-generated audio, and deepfake videos. This project aims to tackle these challenges by combining multiple AI models into a single intelligent system capable of detecting fake media across various formats.

This project uses a **multimodal approach**, meaning it does not rely on only one type of data. Instead, it analyzes and predicts authenticity from:

* 📰 Text News
* 🖼️ Images
* 🎙️ Audio
* 🎥 Videos

Each module has been trained separately using specialized Machine Learning and Deep Learning techniques optimized for its respective media type.

The system demonstrates how AI can be used to combat misinformation, improve media verification, and contribute toward safer digital communication.

---

# 🚀 Problem Statement

With the rise of social media and AI-generated content, fake media has become increasingly difficult to identify manually. Traditional detection systems usually focus on a single modality such as text or image only.

However, real-world misinformation spreads through multiple formats simultaneously.

Examples include:

* Fake news articles
* Edited or AI-generated images
* Voice cloning and synthetic speech
* Deepfake face-swapped videos

This project addresses these problems using a **single multimodal framework** capable of detecting fake content from multiple sources.

---

# 🎯 Objectives

The main objectives of this project are:

* Detect fake and real news articles
* Identify manipulated or AI-generated images
* Classify fake and real audio
* Detect deepfake or manipulated videos
* Demonstrate multimodal AI integration
* Provide an intelligent and scalable misinformation detection framework

---

# 🧠 Project Modules

The system consists of four major modules:

| Module               | Technology Used  | Purpose                   |
| -------------------- | ---------------- | ------------------------- |
| Fake News Detection  | Machine Learning | Detect fake textual news  |
| Fake Image Detection | Deep Learning    | Detect manipulated images |
| Fake Audio Detection | Machine Learning | Detect synthetic audio    |
| Fake Video Detection | Deep Learning    | Detect deepfake videos    |

---
### 📊 About Datasets 

Due to GitHub file size limitations, the audio, image, and video datasets are not uploaded to this repository.
You can download the datasets directly from the following Kaggle links and place them in the appropriate folders before running the project.

# 📂 Module 1: Fake News Detection

## Overview

The Fake News Detection module classifies news articles or statements as **Real** or **Fake** using Natural Language Processing and Machine Learning.

This model combines multiple datasets and applies extensive text preprocessing before classification.

---

## Dataset Used

This module combines:

### 1. True and Fake News Dataset

Contains real and fake news articles.

### 2. LIAR Dataset

Contains political statements with truthfulness labels.

The datasets are merged to improve diversity and model generalization.

---

## ⚙️ Text Preprocessing

The following preprocessing steps are performed:

### Text Cleaning

* Lowercasing
* URL removal
* HTML tag removal
* Punctuation removal
* Number removal

### NLP Processing

* Stopword removal
* Tokenization
* Lemmatization

These steps help improve model accuracy by reducing noise.

---

## Feature Engineering

The model uses:

### TF-IDF Vectorization

TF-IDF converts text into numerical vectors representing word importance.

Configuration:

* Maximum Features: 10000
* N-Gram Range: (1,2)
* Minimum Document Frequency: 2

This allows the model to capture both single words and phrases.

---

## Machine Learning Model

The classifier used is:

### Logistic Regression

Reasons for selection:

* Fast training
* High interpretability
* Good NLP performance
* Efficient for sparse TF-IDF features

---

## Output

The module predicts:

* Real News ✅
* Fake News ❌

---

# 🖼️ Module 2: Fake Image Detection

## Overview

The Fake Image Detection module uses **Deep Learning and Transfer Learning** to classify images as real or fake.

The system uses pretrained CNN architecture for high accuracy and better feature extraction.

---

## Dataset Structure

image dataset link - https://www.kaggle.com/datasets/duyminhle/real-fake-image-dataset?utm_source=chatgpt.com

The model uses image folders:

```text
IMAGES/
    REAL/
    FAKE/
```

Images are loaded and processed automatically.

---

## Image Processing

Images are:

* Resized to 224×224
* Preprocessed
* Augmented

---

## Data Augmentation

The following augmentation techniques are applied:

* Rotation
* Zoom
* Horizontal Flip
* Shearing
* Brightness Adjustment
* Width Shift
* Height Shift

Benefits:

* Reduces overfitting
* Improves generalization
* Creates robust training data

---

## Deep Learning Architecture

The image model uses:

# MobileNetV2

Transfer learning is used with pretrained ImageNet weights.

Configuration:

* Include Top = False
* Fine-Tuning Enabled
* Custom Dense Layers Added

Architecture includes:

* GlobalAveragePooling
* BatchNormalization
* Dense Layers
* Dropout Layers
* Sigmoid Output Layer

---

## Training Optimization

Callbacks used:

### EarlyStopping

Stops training when performance stops improving.

### ReduceLROnPlateau

Automatically reduces learning rate.

### ModelCheckpoint

Saves best-performing model.

---

## Output

Predictions:

* Real Image ✅
* Fake Image ❌

---

# 🎙️ Module 3: Fake Audio Detection

## Overview

The Fake Audio Detection module detects synthetic or manipulated voice recordings.

It uses **audio feature engineering** combined with Machine Learning classifiers.

---

## Dataset Structure

Audio dataset link - audio dataset link - https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition?select=KAGGLE

```text
AUDIO/
    REAL/
    FAKE/
```

Supported format:

```text
.wav
```

---

## Audio Feature Extraction

Audio files are processed using:

# Librosa

The system extracts **59 audio features**.

---

## Extracted Features

### MFCC

Captures speech characteristics.

### Spectral Centroid

Measures audio brightness.

### Spectral Bandwidth

Represents frequency spread.

### Spectral Rolloff

Frequency distribution indicator.

### Spectral Flatness

Noise measurement.

### Zero Crossing Rate

Signal sign changes.

### RMS Energy

Audio intensity.

### Chroma Features

Pitch distribution information.

---

## Feature Scaling

Features are normalized using:

### StandardScaler

Benefits:

* Improved convergence
* Stable model performance

---

## Machine Learning Models

The module uses:

### Random Forest

and

### XGBoost

These are combined using:

### Voting Classifier

Benefits:

* Higher accuracy
* Better robustness
* Reduced variance

---

## Evaluation Metrics

The audio model uses:

* Accuracy
* Classification Report
* Confusion Matrix
* ROC-AUC Score

---

## Output

Predictions:

* Real Audio ✅
* Fake Audio ❌

---

# 🎥 Module 4: Fake Video Detection

## Overview

The Fake Video Detection module identifies deepfake or manipulated videos using Deep Learning.

This module focuses primarily on **face-based manipulation detection**.

---

## Dataset Used

video dataset link - video dataset link - https://www.kaggle.com/datasets/ahmedelbanby/faceforensicsplusplus-c23-deepfakebench-structure?select=videos

The project uses:

### FaceForensics++

One of the most popular deepfake datasets.

Contains:

* Real Videos
* Manipulated Videos

---

## Video Processing Pipeline

The detection pipeline includes:

### Video Loading

Using:

```python
cv2.VideoCapture()
```

---

## Frame Extraction

Frames are sampled from:

* 10% to 90% of video duration

Default:

```text
10 frames per video
```

This balances:

* Accuracy
* Memory usage
* Training speed

---

## Face Detection

The system uses:

### Haar Cascade Classifier

Faces are:

* Detected
* Cropped
* Resized

Fallback:

Center crop if no face detected.

---

## Deep Learning Architecture

The module uses:

### MobileNetV2

Reasons:

* Lightweight
* Efficient
* Fast inference
* Strong transfer learning performance

---

## Deepfake Detection Workflow

```text
Video
   ↓
Frame Extraction
   ↓
Face Detection
   ↓
Face Cropping
   ↓
CNN Feature Extraction
   ↓
Classification
   ↓
Real / Fake
```

---

## Output

Predictions:

* Real Video ✅
* Fake Video ❌

---

# 🛠️ Technologies Used

## Programming Language

* Python

---

## Libraries Used

### Machine Learning

* Scikit-Learn
* XGBoost
* Joblib

### Deep Learning

* TensorFlow
* Keras

### Computer Vision

* OpenCV

### Audio Processing

* Librosa

### Data Handling

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### NLP

* NLTK
* TF-IDF

---

# 📁 Recommended Folder Structure

```text
Project/
│
├── Fake_News_Detection.ipynb
├── Fake_Image_Detection.ipynb
├── Fake_Audio_Detection.ipynb
├── Fake_Video_Detection.ipynb
│
├── AUDIO/
│   ├── REAL/
│   └── FAKE/
│
├── IMAGES/
│   ├── REAL/
│   └── FAKE/
│
├── FaceForensics++/
│
├── models/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚡ Installation

Clone repository:

```bash
git clone <repository-link>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks or application.

---

# 📈 Evaluation Metrics

The project evaluates models using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC-AUC

These metrics provide balanced model assessment.

---

# 🔮 Future Improvements

Possible future enhancements:

* Real-time detection
* Multimodal fusion model
* Transformer-based NLP
* Advanced Deepfake detection
* Cloud deployment
* Streamlit/Web dashboard
* OCR-based image-text verification
* Explainable AI integration

---

# 🌍 Real World Applications

This project can be used in:

* Journalism
* Social Media Monitoring
* Cybersecurity
* News Verification
* Digital Forensics
* Media Authentication
* Educational Research

---

# 📚 Learning Outcomes

This project demonstrates practical understanding of:

### Machine Learning

* Logistic Regression
* Random Forest
* XGBoost

### Deep Learning

* CNN
* Transfer Learning
* MobileNetV2
* Fine-Tuning

### NLP

* Text Cleaning
* TF-IDF
* Lemmatization

### Audio Processing

* MFCC
* Spectral Features

### Computer Vision

* Face Detection
* Image Processing
* Video Analysis

---
# ⚠️ Important Note About Datasets

The datasets are not included in this repository because:

GitHub has file upload size limitations
Deep learning datasets are extremely large
Uploading large datasets makes repositories difficult to manage

After downloading the datasets:

Extract all files properly
Place them in the correct folders
Update dataset paths in notebooks if necessary
Install all required dependencies before training/testing the models

# ⚠️ Disclaimer

This project is developed for **educational and research purposes**. Predictions may vary depending on dataset quality and real-world conditions. The system should be used as an assistive verification tool rather than a final authority.

---

# 👨‍💻 Author

**Nishkarsh**
BCA Student | AI & Data Science Enthusiast | Machine Learning and Deep Learning Learner

---

# ⭐ Support

If you found this project useful:

⭐ Star this repository
🍴 Fork this project
📢 Share with others

---

**"Fighting misinformation through Artificial Intelligence and Multimodal Learning."** 🚀
