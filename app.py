import streamlit as st
import joblib
import cv2
import numpy as np
from skimage.feature import hog
from PIL import Image
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Pet Classifier",
    page_icon="🐾",
    layout="wide"
)

# --- Custom CSS for Animations, Colors, and Modern Look ---
st.markdown("""
<style>
    .stApp {
        background-color: #e9e6ff; /* Soft Lavender */
    }
    .title-text {
        color: #3c3b6e; /* Dark Purple */
        text-align: center;
        font-size: 2.8rem;
        font-weight: bold;
    }
    .subheader-text {
        color: #5a5a8d; /* Muted Purple */
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 6px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 2rem !important;
        font-weight: bold;
        text-align: center;
    }
    .cat-text {
        color: #ff8c00; /* Vibrant Orange for Cat */
    }
    .dog-text {
        color: #1e90ff; /* Dodger Blue for Dog */
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loading and Feature Extraction (HOG+SVM Version) ---
@st.cache_resource
def load_model():
    """Loads the HOG+SVM model."""
    try:
        model = joblib.load('svm_model.joblib')
        return model
    except FileNotFoundError:
        return None

model = load_model()

def extract_features(image):
    """Extracts HOG features for the SVM model."""
    resized_image = cv2.resize(image, (64, 128))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return hog_features.reshape(1, -1)


# --- Main App Title ---
st.markdown('<p class="title-text">Image Classifier: Cats vs. Dogs</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Discover if your pet\'s picture is classified as a Cat 🐱 or a Dog 🐶</p>', unsafe_allow_html=True)

# Check if model exists
if model is None:
    st.error("Model file (`svm_model.joblib`) not found. Please run the final `train_model.py` script first.", icon="🚨")
else:
    # --- Create Tabs ---
    tab1, tab2, tab3 = st.tabs(["🐾 Classifier", "⚙️ How It Works", "📊 About the Model"])

    # --- Classifier Tab ---
    with tab1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image to get started...", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                image = Image.open(uploaded_file)
                st.image(image, caption='Your Uploaded Image', use_column_width='always')
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                with st.spinner('Model is analyzing the image...'):
                    time.sleep(1) 
                    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    features = extract_features(opencv_image)
                    probabilities = model.predict_proba(features)[0]
                    confidence = max(probabilities)
                    prediction = model.classes_[np.argmax(probabilities)]
                
                st.subheader("Classification Result")
                st.divider()

                # --- NEW SIMPLIFIED LOGIC ---
                # No more threshold check. Always show the prediction.
                if prediction == 'Dog':
                    st.markdown('<p class="result-text dog-text">Prediction: It\'s a Dog! 🐶</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="result-text cat-text">Prediction: It\'s a Cat! 🐱</p>', unsafe_allow_html=True)
                
                # Still show the confidence score so the user knows how sure the model is
                st.write("")
                st.progress(confidence)
                st.write(f"**Confidence:** {confidence:.2%}")
                
                st.balloons()
                st.markdown('</div>', unsafe_allow_html=True)

    # --- How It Works Tab ---
    with tab2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("The Two-Step Process")
        st.divider()
        st.markdown("""
        This classifier uses a classic two-step computer vision process:
        1. **Feature Extraction:** The image is converted into a mathematical summary of its shapes and lines (a Histogram of Oriented Gradients or HOG).
        2. **Classification:** A pre-trained Support Vector Machine (SVM) model uses this summary to predict whether the pattern is more like a cat or a dog.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- About the Model Tab ---
    with tab3:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Model Training & Performance")
        st.divider()
        st.markdown("The model was trained on the full Asirra (Cats vs. Dogs) dataset, containing nearly 25,000 images.")
        st.metric(label="Expected Accuracy", value="~85-90%")
        st.info("Training on the full dataset provides a much more robust and accurate model than using a small sample.")
        st.markdown('</div>', unsafe_allow_html=True)