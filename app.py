import os
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from dataset import load_data

MODEL_PATH = "dr_effnet_model.h5"
INPUT_SIZE = (224, 224)
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

data_path = load_data()
st.success(f"Dataset ready at: {data_path}")

@st.cache_resource
def load_my_model():
    model = load_model(MODEL_PATH)
    return model

model = load_my_model()

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(INPUT_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("Retinal Disease Prediction with EfficientNet")
st.write("Upload a fundus image to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_processed = preprocess_image(image)
    predictions = model.predict(img_processed)

    st.write("Raw model output:", predictions)
    st.write("Predictions shape:", predictions.shape)

    pred_index = np.argmax(predictions)
    confidence = predictions.flatten()[pred_index]

    if pred_index < len(CLASS_NAMES):
        pred_class = CLASS_NAMES[pred_index]
    else:
        pred_class = f"Class index {pred_index} not in CLASS_NAMES"

    st.write(f"**Prediction:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
