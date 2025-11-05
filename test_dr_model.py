import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import csv

project_path = r"C:\Users\aswin\Documents\DR_EfficientNet"

dataset_path = os.path.join(project_path, "archive", "grayscale_images")

model_path = os.path.join(project_path, "dr_effnet_model.keras")
output_csv = os.path.join(project_path, "predictions_all.csv")  # new CSV to avoid permission issues

model = load_model(model_path)
print("Model loaded successfully.")

class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']
print("Detected classes:", class_names)

# Gather all image files
all_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_files.append(os.path.join(root, file))

# Open CSV once
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Predicted_Class", "True_Class"])  # header

    for img_path in all_files:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img), axis=0)
        pred = model.predict(img_array, verbose=0)
        predicted_label = class_names[np.argmax(pred)]
        true_label = os.path.basename(os.path.dirname(img_path))
        writer.writerow([os.path.basename(img_path), predicted_label, true_label])

print(f"Predictions for all images saved to: {output_csv}")
