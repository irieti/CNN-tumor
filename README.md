# Brain Tumor MRI Classification - Clean Implementation

## 📋 Overview

This is a streamlined CNN implementation for classifying brain MRI images into four categories:
- **Glioma** - Tumor in the brain/spinal cord
- **Meningioma** - Tumor in the meninges
- **No Tumor** - Healthy brain scans
- **Pituitary** - Tumor in the pituitary gland


## 🏗️ Model Architecture

```
Input (224×224×3 RGB image)
    ↓
Conv Block 1: 3→32 channels, 224→112 spatial (MaxPool)
    ↓
Conv Block 2: 32→64 channels, 112→56 spatial (MaxPool)
    ↓
Conv Block 3: 64→128 channels, 56→28 spatial (MaxPool)
    ↓
Conv Block 4: 128→256 channels, 28→14 spatial (MaxPool)
    ↓
Flatten: 256×14×14 = 50,176 features
    ↓
FC Layer 1: 50,176→512 neurons (ReLU + Dropout 50%)
    ↓
FC Layer 2: 512→4 classes (output logits)
```

**Each Conv Block contains:**
- Conv2d (3×3 kernel, padding=1)
- BatchNorm2d
- ReLU activation
- MaxPool2d (2×2)
- Dropout (25%)

**Total Parameters:** ~25.8 million

## 📦 Installation

### Requirements

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install torch torchvision numpy pandas matplotlib Pillow scikit-learn seaborn tqdm
```

### Dataset Setup

1. Download from Kaggle:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

2. Update paths in code:
```python
TRAIN_DIR = '/path/to/archive/Training'
TEST_DIR = '/path/to/archive/Testing'
```

3. Verify structure:
```
archive/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```# CNN
# CNN-tumor
