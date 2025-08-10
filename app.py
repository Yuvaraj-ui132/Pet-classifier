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
    /* Keyframe animation for fade-in effect */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Main app background with modern purple shade */
    .stApp {
        background-color: #e9e6ff; /* Soft Lavender */
    }

    /* Custom title style with animation */
    .title-text {
        color: #3c3b6e; /* Dark Purple */
        text-align: center;
        font-size: 2.8rem;
        font-weight: bold;
        animation: fadeIn 1s ease-out;
    }
    
    /* Custom subheader style with animation */
    .subheader-text {
        color: #5a5a8d; /* Muted Purple */
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1.2s ease-out;
    }

    /* Styling for info cards with animation and hover effect */
    .info-box {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 6px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 1.5s ease-out;
        margin-bottom: 1rem;
    }
    .info-box:hover {
        transform: translateY(-5px); /* Lift card on hover */
        box-shadow: 0 12px 20px rgba(90,90,141,0.2); /* Glow effect */
    }

    /* Result text style */
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


# --- Model Loading and Feature Extraction (No changes needed here) ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('svm_model.joblib')
        return model
    except FileNotFoundError:
        return None

model = load_model()

def extract_features(image):
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
    st.error("Model file (`svm_model.joblib`) not found. Please run the `train_model.py` script first.", icon="🚨")
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
                    time.sleep(1) # Small delay for better UX
                    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    features = extract_features(opencv_image)
                    prediction = model.predict(features)
                
                st.subheader("Classification Result")
                st.divider()
                if prediction[0] == 'Dog':
                    st.markdown('<p class="result-text dog-text">Prediction: It\'s a Dog! 🐶</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="result-text cat-text">Prediction: It\'s a Cat! 🐱</p>', unsafe_allow_html=True)
                st.balloons()
                st.markdown('</div>', unsafe_allow_html=True)

    # --- How It Works Tab ---
    with tab2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("The Two-Step Process")
        st.divider()
        st.markdown("""
        This classifier doesn't "see" images like we do. Instead, it uses a classic two-step computer vision process:

        #### 1. Feature Extraction
        First, the image is converted into a mathematical summary called a **Histogram of Oriented Gradients (HOG)**. This process breaks the image into small cells and analyzes the direction of edges and gradients within them. This effectively describes the basic shapes (like pointy ears or a round snout) as a vector of numbers.

        #### 2. Classification
        Second, this HOG vector is fed into the pre-trained **Support Vector Machine (SVM)** model. The SVM's job is to find a dividing line (a "hyperplane") that best separates the numerical patterns of "cat" HOGs from "dog" HOGs based on what it learned during training. Your image's features fall on one side of this line, leading to the final prediction.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- About the Model Tab ---
    with tab3:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.subheader("Model Training & Performance")
        st.divider()
        st.markdown("""
        The model used in this application was trained on a small, random sample of the **Asirra (Cats vs. Dogs) dataset** from Microsoft Research.
        """)
        st.metric(label="Images in Training Sample", value="2,000")
        st.metric(label="Accuracy on Test Data", value="64.00 %", delta="-25-30% vs. full dataset", delta_color="inverse")
        st.warning("This model was intentionally trained on a small dataset for rapid demonstration. A model trained on the full dataset of 25,000 images would achieve much higher accuracy but would take significantly longer to prepare.", icon="⚠️")
        st.markdown('</div>', unsafe_allow_html=True)