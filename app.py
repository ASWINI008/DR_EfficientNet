import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

MODEL_PATH = "dr_effnet_model.h5"
INPUT_SIZE = (224, 224)
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

# Mapping 16-class model output to 5-class categories

CLASS_MAPPING = {
0: 0, 1: 0,                 # No DR
2: 1,                        # Mild DR
3: 2, 4: 2,                  # Moderate DR
5: 3, 6: 3, 7: 3,            # Severe DR
8: 4, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4  # Proliferative DR
}

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()
    return load_model(MODEL_PATH)

model = load_my_model()

def preprocess_image(image: Image.Image):
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

with st.spinner("Predicting..."):
    predictions = model.predict(img_processed)

# Aggregate predictions for 5 classes
pred_probs_5 = np.zeros(len(CLASS_NAMES))
for idx_16, idx_5 in CLASS_MAPPING.items():
    pred_probs_5[idx_5] += predictions[0][idx_16]

pred_index_5 = int(np.argmax(pred_probs_5))
pred_class_5 = CLASS_NAMES[pred_index_5]
confidence_5 = pred_probs_5[pred_index_5]

st.markdown(f"### 🩺 Prediction: **{pred_class_5}**")
st.write(f"**Confidence:** {confidence_5 * 100:.2f}%")


