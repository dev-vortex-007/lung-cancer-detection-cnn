import os
import keras
from keras import layers, models
import matplotlib.pyplot as plt

# Paths
base_dir = os.path.join(os.path.dirname(__file__), '../data/raw/kaggle-chest-ct-scan-images-dataset')

# Image parameters
IMG_SIZE = (400, 400)
BATCH_SIZE = 32

# Data preprocessing
train_ds = keras.utils.image_dataset_from_directory(
    f"{base_dir}/train",            # your train folder path
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = keras.utils.image_dataset_from_directory(
    f"{base_dir}/valid",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_ds = keras.utils.image_dataset_from_directory(
    f"{base_dir}/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Model definition (simple CNN)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 classes with softmax
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # For multi-class classification
              metrics=['accuracy'])

# Summary
model.summary()

# Training
EPOCHS = 16
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save model
os.makedirs("../models", exist_ok=True)
model.save("../models/cnn_model.keras")

# Plot results
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.show()
