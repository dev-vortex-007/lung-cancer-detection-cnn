## Overview

This repository contains a complete Lung Cancer Detection system using Convolutional Neural Networks (CNN) for analyzing chest CT scan images. The system includes both training capabilities and multiple prediction interfaces.

**Features:**
- 🧠 CNN model training with chest CT scan images
- 🖥️ Beautiful GUI interface for easy image analysis
- 💻 Command-line interface for batch processing
- 🚀 One-click launcher for easy system access
- 📊 Detailed prediction results with confidence scores

This README explains the training process, prediction system, and how to use all the available tools.

## File: `src/train.py` — high-level summary

`train.py` builds and trains a small CNN using Keras (TensorFlow backend). The script:
- Loads images from a directory using `ImageDataGenerator` and `flow_from_directory`.
- Applies image preprocessing and augmentation.
- Defines a sequential CNN with a few Conv2D and MaxPooling layers, followed by Dense layers.
- Trains the model for a fixed number of epochs and saves the trained model to `models/cnn_model.h5`.
- Plots training and validation accuracy.

## Expected project structure and data layout

The script expects a `data/` folder at the repository root (relative paths in `train.py` use `../data` from `src/`). The dataset structure should be in the Keras `flow_from_directory` format, for example:

data/
  train/
    adenocarcinoma/
    large_cell_carcinoma/
    normal/
    squamous_cell_carcinoma/
  test/   (optional — not used by this script)

Note: `train.py` uses `flow_from_directory(train_dir, subset='training')` and `subset='validation'` with `validation_split=0.2`. That means all images must be inside the `train/` directory in per-class subfolders.

## Line-by-line/section-by-section explanation and parameters

- Imports and setup
  - `import os`
  - `import tensorflow as tf` — TensorFlow provides the backend and Keras APIs used.
  - `from keras.preprocessing.image import ImageDataGenerator` — image loading and augmentation utilities.
  - `from keras import layers, models` — Keras model building blocks.
  - `import matplotlib.pyplot as plt` — plotting training history.

- Paths
  - `base_dir = os.path.join(os.path.dirname(__file__), '../data')`
    - This resolves `data/` relative to the `src/` folder. If you run `src/train.py`, `base_dir` becomes `../data` from the `src` directory.
  - `train_dir = os.path.join(base_dir, 'train')`
  - `test_dir = os.path.join(base_dir, 'test')` (not used in the training call below; present for possible evaluation use)

- Image parameters
  - `IMG_SIZE = (224, 224)`
    - Target image size used by `flow_from_directory`. Each image will be resized to 224x224 pixels. 224x224 is a common size for CNNs (e.g., models pretrained on ImageNet). Larger sizes increase memory and compute.
  - `BATCH_SIZE = 32`
    - Number of images per training batch. A power of two like 16/32/64 is common. If you run out of GPU memory, reduce batch size.

- Data preprocessing and augmentation (`ImageDataGenerator` parameters)
  - `rescale=1./255`
    - Scales pixel values from [0,255] to [0.0, 1.0]. This is required when using float-based model inputs and common for neural nets.
  - `rotation_range=10`
    - Randomly rotates images by up to +/-10 degrees. Helps the model generalize to rotated inputs.
  - `width_shift_range=0.1` and `height_shift_range=0.1`
    - Random horizontal/vertical shifts up to 10% of the image width/height.
  - `shear_range=0.1`
    - Shear transformation range. Minor geometric distortions to augment data variability.
  - `zoom_range=0.1`
    - Random zoom within +/-10%.
  - `horizontal_flip=True`
    - Random horizontal flips. Be careful: flipping may or may not be appropriate depending on the medical imaging context (for bilateral organs like lungs it's usually acceptable, but consult domain knowledge).
  - `validation_split=0.2`
    - Reserves 20% of images from `train_dir` for validation. `flow_from_directory` will use `subset='training'` and `subset='validation'` to split.

- Directory iterators
  - `train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', subset='training')`
  - `val_gen = ... subset='validation'`

  - Important: `class_mode='binary'` tells Keras the task is a binary classification problem and it will produce a single label per sample (0 or 1) and expect the model to output a single probability (sigmoid). However, the dataset folders shown in the repository (adenocarcinoma, large_cell_carcinoma, normal, squamous_cell_carcinoma) imply 4 classes. If you have 4 classes you should use `class_mode='categorical'`, make the final Dense layer `Dense(4, activation='softmax')`, and use `loss='categorical_crossentropy'`.

  - If you truly have a binary problem (e.g., `normal` vs `cancer`), ensure your directory contains exactly two class subfolders and keep `class_mode='binary'`.

- Model definition (simple CNN)
  - Sequential model with layers:
    - Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3))
      - 32 filters, 3x3 kernel, ReLU activation. `input_shape=(224,224,3)` expects RGB images.
    - MaxPooling2D(2,2)
    - Conv2D(64, (3,3), activation='relu')
    - MaxPooling2D(2,2)
    - Conv2D(128, (3,3), activation='relu')
    - MaxPooling2D(2,2)
    - Flatten()
    - Dense(128, activation='relu')
    - Dropout(0.5)
      - Dropout=50% to reduce overfitting.
    - Dense(1, activation='sigmoid')
      - Single output + sigmoid — appropriate for binary classification.

  - `model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])`
    - `optimizer='adam'` — adaptive optimizer, good default.
    - `loss='binary_crossentropy'` — appropriate for binary classification with a single sigmoid output. For multi-class, use `categorical_crossentropy` (with softmax output) or `sparse_categorical_crossentropy` depending on label encoding.

- Training
  - `EPOCHS = 16` — number of full passes over the training dataset.
  - `history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)`
    - Keras will iterate over `train_gen` and evaluate on `val_gen` each epoch. `train_gen` yields batches indefinitely; Keras uses `steps_per_epoch` auto-calculated from generator and batch size unless overridden.

- Saving and plotting
  - `os.makedirs("../models", exist_ok=True)` and `model.save("../models/cnn_model.h5")`
    - Saves model in HDF5 format to repository `models` folder (path is relative to `src/`, creating `../models`).
  - The script plots training and validation accuracy using Matplotlib and calls `plt.show()` (this blocks if running in a non-interactive environment; when running remotely you might want to save the figure instead with `plt.savefig(...)`).

## Inputs and outputs (contract)

- Inputs
  - Directory `data/train/` with subfolders per class containing image files. Images are read and resized to `IMG_SIZE`.
  - Hyperparameters (inside the script): `IMG_SIZE`, `BATCH_SIZE`, augmentation params, `EPOCHS`.

- Outputs
  - Trained model saved to `models/cnn_model.h5`.
  - Displayed Matplotlib accuracy plot (or saved if you modify the script).

## Notable assumptions and mismatch warnings

- class_mode vs dataset classes: The script uses `class_mode='binary'` and a single-output sigmoid. If your data has more than 2 classes (as folder names suggest), switch to `class_mode='categorical'` and change the final Dense to match number of classes with `softmax` activation and use `loss='categorical_crossentropy'`.

- Input channels: Model expects 3-channel color images. If your images are grayscale, convert them to RGB (e.g., repeat channels) or change `input_shape` and data preprocessing accordingly.

- File paths: The script resolves `data` and `models` paths relative to `src/`. Run it from the repository root as `python src\\train.py` (Windows `cmd.exe`) or from `src` but be mindful of relative paths.

## Edge cases and pitfalls

- Class imbalance: If some classes have many more images than others, accuracy can be misleading. Consider class weights or sampling strategies, and monitor per-class metrics (precision/recall/F1).
- Small dataset: The simple CNN may overfit. Consider transfer learning (pretrained models like MobileNet, ResNet) with fine-tuning.
- Running without GPU: Training on CPU will be slow. If you have a GPU, ensure TensorFlow is installed with GPU support.
- Memory/GPU OOM: Reduce `BATCH_SIZE` or image size if you run out of memory.
- plt.show() in headless environments: Use `plt.savefig('training.png')` instead.

## How to run

1. Create a Python environment and install requirements.
   - The repository has a `requirements.txt`. Typically on Windows `cmd.exe`:

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Ensure the `data/train/` folder exists and contains class subfolders as described above.

3. Run training from the repository root (so relative paths line up):

```bash
python src\\train.py
```

4. After training you will find the saved model at `models/cnn_model.h5` and a plot will be shown in a window.

## Suggested improvements (small, low-risk)

- Fix class-mode mismatch: change `class_mode` and final layer if dataset is multi-class.
- Add ModelCheckpoint and EarlyStopping callbacks to save best model and avoid overfitting:
  - `tf.keras.callbacks.ModelCheckpoint('../models/best_model.h5', save_best_only=True)`
  - `tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)`
- Use `steps_per_epoch` and `validation_steps` explicitly when dataset sizes are known.
- Save training plot to file in addition to (or instead of) `plt.show()`.
- Consider using `tf.data` pipelines for better performance on large datasets.

## Quick checklist for common fixes

- If you have 4 classes, update these lines:
  - In data generator calls: `class_mode='categorical'`
  - Final layer: `layers.Dense(4, activation='softmax')`
  - Compile loss: `loss='categorical_crossentropy'`

- If images are grayscale:
  - Convert them to RGB or change `input_shape` to `(IMG_SIZE[0], IMG_SIZE[1], 1)` and ensure `flow_from_directory(..., color_mode='grayscale')`.

## Prediction System

After training your model, you can use the comprehensive prediction system to analyze new CT scan images.

### 🚀 Quick Start (Recommended)

The easiest way to use the system is through the launcher:

```bash
python launcher.py
```

This will:
- Check all requirements
- Verify your trained model exists
- Let you choose between GUI or CLI interface
- Optionally train the model if it doesn't exist

### 🖥️ GUI Interface

For a user-friendly experience with visual feedback:

```bash
python launcher.py --gui
# or directly:
python src/predict.py
```

**GUI Features:**
- 📁 Easy file selection with preview
- 🔍 Real-time image analysis
- 📊 Detailed results with confidence scores
- 🎨 Clean, professional interface
- ⚡ Threaded processing (non-blocking)

### 💻 Command Line Interface

For batch processing and automation:

```bash
python launcher.py --cli
# or directly:
python src/predict_cli.py

# CLI Options:
python src/predict_cli.py --image path/to/image.jpg --show-image
python src/predict_cli.py --directory path/to/images/
python src/predict_cli.py --model custom/model/path.keras
```

**CLI Features:**
- 🔄 Batch processing of multiple images
- 📈 Detailed probability breakdown
- 🖼️ Optional image display
- 📋 Summary reports for batch analysis

### 📊 Prediction Classes

The system can detect four types of lung conditions:
- **Adenocarcinoma** - A type of lung cancer
- **Large Cell Carcinoma** - Another type of lung cancer  
- **Normal** - Healthy lung tissue
- **Squamous Cell Carcinoma** - A type of lung cancer

### 🛠️ System Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow
- opencv-python
- numpy
- matplotlib
- pillow
- pandas
- scikit-learn

### 📝 Usage Examples

**Single Image Analysis (GUI):**
1. Run `python launcher.py`
2. Choose GUI option
3. Click "Select Image"
4. Choose your CT scan image
5. Click "Analyze Image"
6. View detailed results

**Batch Processing (CLI):**
```bash
python src/predict_cli.py --directory ./test_images/
```

**Interactive Mode:**
```bash
python src/predict_cli.py
# Follow the prompts for interactive analysis
```

## Completion Summary

This project provides a complete lung cancer detection pipeline with both training and prediction capabilities. The prediction system offers multiple interfaces to suit different use cases, from quick single-image analysis to automated batch processing.

- Use `class_mode='categorical'` and a 4-unit softmax output (if dataset is multi-class),
- Add ModelCheckpoint and EarlyStopping callbacks,
- Save training plots to a file instead of opening a GUI window.

If you'd like any of those automated changes applied, tell me which and I'll implement them.
