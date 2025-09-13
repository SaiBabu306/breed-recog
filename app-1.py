import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# ===========================
# Download and Load Model
# ===========================
MODEL_PATH = "trained_model.keras"
FILE_ID = "1dQg1WgipCUuMDym1OKew0LQv9ADpQJz3"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive... please wait ⏳")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False, fuzzy=True)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# ===========================
# Cattle Breed Classes
# ===========================
class_names = ["Gir", "Murrah", "Sahiwal"]

# ===========================
# Prediction Function
# ===========================
def model_prediction(test_image):
    """Predict cattle breed from uploaded image."""
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0) / 255.0  # normalize
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions)  # index + confidence

# ===========================
# Sidebar
# ===========================
st.sidebar.title("🐄 Cattle Breed Recognition")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Breed Recognition"])

# ===========================
# Home Page
# ===========================
if app_mode == "Home":
    st.header("CATTLE BREED RECOGNITION SYSTEM")
    try:
        st.image("home_page.jpeg", use_column_width=True)
    except:
        st.info("Upload `home_page.jpeg` in your repo to display a banner image.")

    st.markdown("""
    Welcome to the **Cattle Breed Recognition System**! 🐮🔍

    Upload a cattle image, and our system will identify the breed.  
    Currently, the model supports **3 breeds**:
    - Gir 🐂  
    - Murrah 🐃  
    - Sahiwal 🐄  

    **How to use:**
    1. Go to the **Breed Recognition** page.
    2. Upload an image of the cattle.
    3. Click **Predict** to see the breed and confidence.
    """)

# ===========================
# About Page
# ===========================
elif app_mode == "About":
    st.header("📌 About the Project")
    st.markdown("""
    This project uses a **Convolutional Neural Network (CNN)** trained on cattle images  
    to classify them into **3 breeds: Gir, Murrah, and Sahiwal**.

    #### Dataset Info:
    - Training Images: Gir, Murrah, Sahiwal  
    - Validation Images: Used for tuning  
    - Test Images: Used for final evaluation  

    #### Why This Project?
    Breed recognition can help in:
    - Livestock management  
    - Automated cattle identification  
    - Supporting farmers and the dairy industry  
    """)

# ===========================
# Breed Recognition Page
# ===========================
elif app_mode == "Breed Recognition":
    st.header("🐄 Breed Recognition")
    test_image = st.file_uploader("Upload a Cattle Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("🔎 Analyzing image...")

            result_index, confidence = model_prediction(test_image)

            st.success(f"✅ Prediction: **{class_names[result_index]}**")
            st.info(f"Model Confidence: {confidence*100:.2f}%")
