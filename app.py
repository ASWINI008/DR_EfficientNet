import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

MODEL_PATH = "dr_effnet_model.h5"
INPUT_SIZE = (224, 224)
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

CLASS_MAPPING = {
    0: 0, 1: 0,
    2: 1,
    3: 2, 4: 2,
    5: 3, 6: 3, 7: 3,
    8: 4, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4
}

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return load_model(MODEL_PATH)

model = load_my_model()

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(INPUT_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

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

    img_processed = preprocess_image(image)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            predictions = model.predict(img_processed)
    
        # Aggregate predictions for 5 classes
        pred_probs_5 = np.zeros(len(CLASS_NAMES))
        for idx_16, idx_5 in CLASS_MAPPING.items():
            pred_probs_5[idx_5] += predictions[0][idx_16]
    
        # ✅ Normalize the probabilities (important)
        pred_probs_5 = pred_probs_5 / np.sum(pred_probs_5)
    
        # Get class with highest confidence
        pred_index_5 = int(np.argmax(pred_probs_5))
        pred_class_5 = CLASS_NAMES[pred_index_5]
        confidence_5 = pred_probs_5[pred_index_5]
    
        st.markdown(f"### 🩺 Prediction: **{pred_class_5}**")
        st.write(f"**Confidence:** {confidence_5 * 100:.2f}%")
    
        # Show detailed probabilities
        st.write("#### Detailed Probabilities:")
        for i, cls in enumerate(CLASS_NAMES):
            st.write(f"{cls}: {pred_probs_5[i] * 100:.2f}%")
    
        # ✅ (Optional) Add a "Show Highest" Button
        if st.button("Show Highest Probability Class"):
            st.success(f"The highest predicted class is **{pred_class_5}** with {confidence_5*100:.2f}% confidence!")
    
    
    


