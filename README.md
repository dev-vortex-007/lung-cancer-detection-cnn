# Lung Cancer Detection CNN

A complete Lung Cancer Detection system using Convolutional Neural Networks (CNN) for analyzing chest CT scan images.

**Features:**
- 🧠 CNN model training with chest CT scan images
- 🖥️ Beautiful GUI interface for easy image analysis
- 💻 Command-line interface for batch processing
- 🚀 One-click launcher for easy system access
- 📊 Detailed prediction results with confidence scores

### 1. Setup Environment

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

Download the pre-trained model from Google Drive:
- **Model Link**: https://drive.google.com/file/d/1WbvOGR6hc1XapHQyWMqt_HI8jU_QL4b6/view?usp=sharing
- **Save as**: `models/cnn_model.keras` or `models/cnn_model.h5`

**Option A: Launcher (Recommended)**
```bash
python launcher.py
```

**Option B: GUI Interface**
```bash
python src\\predict.py
```

**Option C: Command Line Interface**
```bash
python src\\predict_cli.py --image path\\to\\image.jpg --show-image
python src\\predict_cli.py --directory path\\to\\images\\
```

## Training Your Own Model

If you want to train a new model with your own data:

### Data Structure
Create a `data/train/` folder with subfolders for each class:
```
data/
  train/
    adenocarcinoma/
    large_cell_carcinoma/
    normal/
    squamous_cell_carcinoma/
```

### Run Training
```bash
python src\\train.py
```

### Single Image Analysis (GUI)
1. Run `python launcher.py`
2. Choose GUI option
3. Click "Select Image"
4. Choose your CT scan image
5. Click "Analyze Image"
6. View detailed results

### Batch Processing (CLI)
```bash
python src\\predict_cli.py --directory ./test_images/
```

## Prediction Classes

The system can detect four types of lung conditions:
- **Adenocarcinoma** - A type of lung cancer
- **Large Cell Carcinoma** - Another type of lung cancer  
- **Normal** - Healthy lung tissue
- **Squamous Cell Carcinoma** - A type of lung cancer

## System Requirements

The following packages are required (install with `pip install -r requirements.txt`):
- tensorflow
- opencv-python
- numpy
- matplotlib
- pillow
- pandas
- scikit-learn
