import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

project_path = r"C:\Users\aswin\Documents\DR_EfficientNet"
dataset_path = os.path.join(project_path, "archive", "grayscale_images")

model_save_path = os.path.join(project_path, "dr_effnet_model.keras")

class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']

num_classes = len(class_names)

existing_folders = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
print("Detected classes in folders:", existing_folders)
for c in class_names:
    if c not in existing_folders:
        raise ValueError(f"Missing class folder: {c}. Please fix dataset folders before training.")

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='validation'
)

base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[checkpoint, early_stop])

print(f"Training complete. Model saved at: {model_save_path}")
