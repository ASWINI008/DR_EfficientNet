# DR_EfficientNet Project

A **Deep Learning project** for **Diabetic Retinopathy (DR) detection** from retinal fundus images using **EfficientNet**. The project includes training, testing, evaluation scripts, and a **Streamlit web app** for interactive predictions.

---

## Project Setup

### 1️⃣ Install Dependencies

Make sure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Dataset

The dataset is **not included due to size constraints**. Download and extract it using:

```bash
python download_dataset.py
```

Follow the instructions in the script to place the dataset in the correct folder.

---

### 3️⃣ Training the Model

To train the model from scratch:

```bash
python train_dr_model.py
```

---

### 4️⃣ Testing on Images

Run predictions on test images:

```bash
python test_dr_model.py
```

---

### 5️⃣ Evaluate Predictions

Evaluate your model’s performance:

```bash
python evaluate_predictions.py
```

---

## Retinal Disease Prediction Web App

This **Streamlit web app** allows you to upload retinal fundus images and predict **diabetic retinopathy stages** using the trained EfficientNet model.

---

### Live Demo

Try the app online: [🔗 Open the App](https://drefficientnet-a6muojwjx5gj8qggmxqr83.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drefficientnet-a6muojwjx5gj8qggmxqr83.streamlit.app/)

---

### Features

* Upload retinal fundus images (`.jpg`, `.jpeg`, `.png`)
* Predict DR stage:

  * No DR
  * Mild DR
  * Moderate DR
  * Severe DR
  * Proliferative DR
* Show confidence percentages and detailed probabilities
* Interactive, easy-to-use interface

---

### Run Locally (Optional)

1. Clone the repository:

```bash
git clone https://github.com/aswini008/DR_EfficientNet.git
cd DR_EfficientNet
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your trained model file `dr_effnet_model.h5` in the project folder.

4. Run the Streamlit app:

```bash
streamlit run main/app.py
```

5. Open your browser at `http://localhost:8501`.

---

### Notes

* Ensure the dataset is downloaded and properly organized before training or testing.
* For deployment and sharing, you can use **Streamlit Cloud** for public access to your app.
* Screenshots, videos, or GIFs of the app can enhance your README for better understanding.

---

This README now **covers everything**: dataset instructions, model training/testing, app usage, and live demo links, making it professional and easy to follow.

---

If you want, I can **also add a section with sample output images/screenshots embedded** so the README visually demonstrates your app predictions.

Do you want me to do that?


