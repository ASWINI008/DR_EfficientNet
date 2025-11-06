import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

MODEL_PATH = "dr_effnet_model.h5"
INPUT_SIZE = (224, 224)
CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

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
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_processed = preprocess_image(image)
    predictions = model.predict(img_processed)

    pred_index_16 = int(np.argmax(predictions))
    pred_index_5 = CLASS_MAPPING.get(pred_index_16, None)

    if pred_index_5 is not None:
        pred_class = CLASS_NAMES[pred_index_5]
        confidence = float(predictions[0][pred_index_16])
        st.markdown(f"### 🩺 Prediction: **{pred_class}**")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
    else:
        st.error(f"Unexpected prediction index: {pred_index_16}")
        st.write("Raw model output:", predictions)


