import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = "dr_effnet_model.h5"
INPUT_SIZE = (224, 224)
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
# Mapping from 16 possible output indices of the model to the 5 desired classes
CLASS_MAPPING = {
    0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 
    8: 4, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 
    15: 4
}

# --- Model Loading ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return load_model(MODEL_PATH)

model = load_my_model()

# --- Image Preprocessing Function ---
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(INPUT_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Streamlit Interface ---
st.title("👁️ Retinal Disease Prediction with EfficientNet")
st.write("Upload a retinal fundus image to detect diabetic retinopathy stages.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    except UnidentifiedImageError:
        st.error("Cannot open this file. Please upload a valid image.")
        st.stop()
    
    # --- Prediction ---
    img_processed = preprocess_image(image)
    with st.spinner("Predicting..."):
        predictions = model.predict(img_processed)
        
        # Aggregate predictions for 5 classes
        pred_probs_5 = np.zeros(len(CLASS_NAMES))
        # Iterate over the model's 16 outputs and sum their probabilities into the 5 target classes
        for idx_16, idx_5 in CLASS_MAPPING.items():
            pred_probs_5[idx_5] += predictions[0][idx_16]
        
        # Normalize the resulting probabilities so they sum to 1
        pred_probs_5 /= pred_probs_5.sum()

        pred_index_5 = int(np.argmax(pred_probs_5))
        pred_class_5 = CLASS_NAMES[pred_index_5]
        confidence_5 = pred_probs_5[pred_index_5] * 100

    # --- Display Results ---
    st.markdown(f"**Prediction:** <span style='font-size: 24px; color: #1F77B4;'>{pred_class_5}</span>", unsafe_allow_html=True)
    st.write(f"**Confidence:** {confidence_5:.2f}%")

    st.subheader("Detailed Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        prob = pred_probs_5[i] * 100
        st.progress(prob / 100)
        st.write(f"{class_name}: {prob:.2f}%")



