1. Install dependencies: pip install -r requirements.txt
2. Train model: python train_dr_model.py
3. Run test on images: python test_dr_model.py
4. Evaluate predictions: python evaluate_predictions.py


# DR_EfficientNet Project

## Dataset
The dataset is not included due to size.  
Download and extract it using:

```bash
python download_dataset.py

# Retinal Disease Prediction with EfficientNet

A deep learning web app to classify stages of diabetic retinopathy from retinal fundus images using EfficientNet.

## 🔗 Live Demo
Try it online: [https://drefficientnet-dzymskeef4ktx3gkhs25wi.streamlit.app/](https://drefficientnet-dzymskeef4ktx3gkhs25wi.streamlit.app/)

## 🧠 Project Overview
This project uses a pretrained EfficientNet-based deep neural network (DNN) to detect **5 classes** of diabetic retinopathy:

- No DR  
- Mild DR  
- Moderate DR  
- Severe DR  
- Proliferative DR  

Users can upload a retinal fundus image via the Streamlit interface to receive an instant prediction along with a confidence score.

## 📁 File Structure


