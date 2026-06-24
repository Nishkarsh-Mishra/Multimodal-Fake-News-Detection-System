import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import librosa
import tempfile
import cv2
import os

st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection System")

# ---------------- Session State ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Helper ----------------
def show_prediction(prediction, confidence=None):
    color = "green" if "REAL" in prediction.upper() else "red"

    st.session_state.history.append({
        "Prediction": prediction,
        "Confidence (%)": round(confidence, 2) if confidence else None
    })

    st.markdown(
        f"<p style='color:{color}; font-weight:bold; font-size:18px;'>{prediction} ({confidence:.2f}%)</p>",
        unsafe_allow_html=True
    )

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    return (
        joblib.load("Models/Text/tfidf_vectorizer.joblib"),
        joblib.load("Models/Text/logistic_regression_model.joblib"),
        tf.keras.models.load_model("Models/Image/fake_image_detector.keras"),
        joblib.load("Models/Audio/audio_scaler.joblib"),
        joblib.load("Models/Audio/audio_model.joblib"),
        # Updated to your fine-tuned model
        tf.keras.models.load_model("Models/Video/Deepfake_MobileNetV2_FineTuned.keras")
    )

text_vectorizer, text_model, image_model, audio_scaler, audio_model, video_model = load_models()

# ---------------- Sidebar ----------------
st.sidebar.title("📌 About")
st.sidebar.write("Multimodal Fake News Detection")

st.sidebar.markdown("""
### 🧠 System Capabilities

📰 **Fake Text Detection**\n 
🖼️ **Fake Image Detection**\n 
🎧 **Fake Audio Detection**\n 
🎥 **Fake Video Detection**
""")

st.sidebar.subheader("History")
for item in st.session_state.history[-5:][::-1]:
    st.sidebar.write(f"{item['Prediction']} ({item['Confidence (%)']}%)")

# ---------------- Image ----------------
def preprocess_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    img = image.load_img(path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = preprocess_input(img)

    return np.expand_dims(img, axis=0)

# ---------------- Audio ----------------
@st.cache_data
def extract_audio_features_cached(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    y, sr = librosa.load(path, sr=22050)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

    features = np.hstack([mfcc, chroma, contrast])
    return features.reshape(1, -1)

# ---------------- Video ----------------
IMG_SIZE = 128
FRAMES_PER_VIDEO = 15

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

FACE_CASCADE = cv2.CascadeClassifier(cascade_path)

if FACE_CASCADE.empty():
    st.error("❌ Haarcascade file not loaded. Put XML file in project folder.")

def crop_face(frame, padding=0.2):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Note: Frame is already converted to RGB before passing here
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    h, w = frame.shape[:2]

    if len(faces) == 0:
        # Fallback: Center crop if no face detected
        margin_h, margin_w = int(h*0.2), int(w*0.2)
        crop = frame[margin_h:h-margin_h, margin_w:w-margin_w]
        return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    pad_x, pad_y = int(fw * padding), int(fh * padding)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(w, x + fw + pad_x), min(h, y + fh + pad_y)

    face_crop = frame[y1:y2, x1:x2]
    return cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))

def predict_video(video_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_file.read())
        path = tmp.name

    cap = cv2.VideoCapture(path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return "Video processing failed", 0

    # Get evenly spaced frames across the video to avoid just looking at the first 1 second
    indices = np.linspace(int(total_frames*0.1), int(total_frames*0.9), FRAMES_PER_VIDEO, dtype=int)

    preds = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue

        # 1. Convert BGR (OpenCV format) to RGB (Model training format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Extract the face! (This was missing in your UI loop)
        face = crop_face(frame)

        # 3. Apply MobileNetV2 Preprocessing
        face_array = np.array(face, dtype="float32")
        face_array = np.expand_dims(face_array, axis=0)  # (1, 128, 128, 3)
        face_array = preprocess_input(face_array)

        pred = video_model.predict(face_array, verbose=0)[0][0]
        preds.append(pred)

    cap.release()

    if len(preds) == 0:
        return "Video processing failed", 0

    avg_pred = float(np.mean(preds))

    # In MobileNet setup: 1 is REAL, 0 is FAKE
    if avg_pred > 0.5:
        return "Video News is REAL", avg_pred * 100
    else:
        return "Video News is FAKE", (1 - avg_pred) * 100

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Text", "Image", "Audio", "Video"])

# ================= TEXT =================
with tab1:
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""

    def clear_text():
        st.session_state.text_input = ""

    text_input = st.text_area("Enter text", key="text_input")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict Text"):
            if text_input.strip():
                features = text_vectorizer.transform([text_input])
                pred = text_model.predict(features)[0]
                proba = text_model.predict_proba(features)[0]
                conf = max(proba) * 100

                show_prediction("Text News is REAL" if pred == 0 else "Text News is FAKE", conf)

    with col2:
        st.button("Clear Text", on_click=clear_text)

# ================= IMAGE =================
with tab2:
    if "clear_image" not in st.session_state:
        st.session_state.clear_image = False

    if st.session_state.clear_image:
        img_file = st.file_uploader("Upload Image", key="img_empty")
        st.session_state.clear_image = False
    else:
        img_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"], key="img")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict Image"):
            if img_file:
                st.image(img_file)

                img = preprocess_image(img_file)
                pred = image_model.predict(img)[0][0]

                prediction = "Image News is REAL" if pred < 0.5 else "Image News is FAKE"
                conf = (1 - pred if pred < 0.5 else pred) * 100

                show_prediction(prediction, conf)

    with col2:
        if st.button("Clear Image"):
            st.session_state.clear_image = True
            st.rerun()

# ================= AUDIO =================
with tab3:
    if "clear_audio" not in st.session_state:
        st.session_state.clear_audio = False

    if st.session_state.clear_audio:
        audio_file = st.file_uploader("Upload Audio", key="audio_empty")
        st.session_state.clear_audio = False
    else:
        audio_file = st.file_uploader("Upload Audio", type=["wav","mp3"], key="audio")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict Audio"):
            if audio_file:
                st.audio(audio_file)

                file_bytes = audio_file.read()
                feat = extract_audio_features_cached(file_bytes)

                scaled = audio_scaler.transform(feat)
                pred = audio_model.predict(scaled)[0]

                if hasattr(audio_model, "predict_proba"):
                    proba = audio_model.predict_proba(scaled)[0]
                    conf = max(proba) * 100
                else:
                    conf = 85.0

                show_prediction(
                    "Audio News is REAL" if pred == 0 else "Audio News is FAKE",
                    conf
                )
            else:
                st.warning("Upload audio file")

    with col2:
        if st.button("Clear Audio"):
            st.session_state.clear_audio = True
            st.rerun()

# ================= VIDEO =================
with tab4:
    if "clear_video" not in st.session_state:
        st.session_state.clear_video = False

    if st.session_state.clear_video:
        video_file = st.file_uploader("Upload Video", key="video_empty")
        st.session_state.clear_video = False
    else:
        video_file = st.file_uploader("Upload Video", type=["mp4","avi"], key="video")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict Video"):
            if video_file:
                st.video(video_file)

                with st.spinner("Processing video..."):
                    prediction, conf = predict_video(video_file)

                show_prediction(prediction, conf)
            else:
                st.warning("Upload video file")

    with col2:
        if st.button("Clear Video"):
            st.session_state.clear_video = True
            st.rerun()

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("<center>Developed by Nishkarsh</center>", unsafe_allow_html=True)
