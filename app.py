import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

MODEL_PATH = "dr_effnet_model.h5"
INPUT_SIZE = (224, 224)

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return load_model(MODEL_PATH)

model = load_my_model()

num_classes = model.output_shape[1]
CLASS_NAMES = [f"Class {i}" for i in range(num_classes)]

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
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    img_processed = preprocess_image(image)
    predictions = model.predict(img_processed)
    st.write("Model output shape:", predictions.shape)
    pred_index = int(np.argmax(predictions))
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(predictions[0][pred_index])
    st.markdown(f"### 🩺 Prediction: **{pred_class}**")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

