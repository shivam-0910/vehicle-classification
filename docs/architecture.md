# Project Architecture

## Overview

This document describes the technical architecture of the Vehicle Image Classification System, including the project structure, data pipeline, model architecture, training process, and inference workflow.

---

## Project Structure

```
vehicle-classification/
├── data/                          # Dataset (raw + processed)
│   ├── raw/                       # Original dataset (not in repo)
│   ├── processed/                 # Preprocessed dataset
│   │   ├── train/                 # Training images (10 classes)
│   │   ├── validation/            # Validation images (10 classes)
│   │   ├── test/                  # Test images (10 classes)
│   │   └── manifest.csv           # Dataset manifest
│   └── README.md                  # Dataset documentation
│
├── src/                           # Source code (all reusable modules)
│   ├── config/                    # Configuration files
│   │   ├── config.yaml            # Baseline configuration
│   │   ├── Config_phase10_5_exp2.yaml  # Experiment 2 config
│   │   └── config_phase10_5_exp3.yaml  # Experiment 3 config (FINAL)
│   │
│   ├── data/                      # Data processing pipeline
│   │   ├── loader.py              # Load config and dataset
│   │   ├── preprocess.py          # Preprocessing (resize, normalize)
│   │   ├── augmentation.py        # Data augmentation
│   │   └── validator.py           # Data validation
│   │
│   ├── models/                    # Model architecture
│   │   └── cnn_model.py           # CNN architecture definition
│   │
│   ├── training/                  # Training pipeline
│   │   └── train.py               # Training loop with callbacks
│   │
│   ├── evaluation/                # Evaluation pipeline
│   │   └── evaluate.py            # Model evaluation & metrics
│   │
│   ├── inference/                 # Inference pipeline (deployment)
│   │   ├── __init__.py            # Package marker
│   │   └── predict.py             # Single-image prediction CLI
│   │
│   └── utils/                     # Utility functions
│       ├── logger.py              # Logging configuration
│       └── helpers.py             # Helper functions
│
├── app/                           # Deployment applications
│   └── streamlit_app.py           # (Reference to app.py)
│
├── models/                        # Trained model checkpoints
│   ├── best_model.keras           # Baseline model
│   ├── best_model_exp2.keras      # Experiment 2 model (preserved)
│   └── best_model_exp3.keras      # Experiment 3 model (FINAL)
│
├── results/                       # Training & evaluation results
│   ├── evaluation_exp3/           # Final evaluation metrics
│   │   ├── test_metrics.json      # Test set metrics
│   │   ├── per_class_metrics.csv  # Per-class precision/recall/F1
│   │   ├── confusion_matrix.csv   # Confusion matrix data
│   │   └── confusion_matrix.png   # Confusion matrix visualization
│   │
│   ├── metrics_exp3/              # Training history
│   │   └── training_history.json  # Loss/accuracy per epoch
│   │
│   ├── plots_exp3/                # Training plots
│   │   ├── training_accuracy.png  # Training accuracy curve
│   │   └── training_loss.png      # Training loss curve
│   │
│   ├── evaluation_exp2/           # Experiment 2 results (archived)
│   ├── metrics_exp2/              # Experiment 2 metrics
│   ├── plots_exp2/                # Experiment 2 plots
│   │
│   └── final_model.json           # Metadata for final model
│
├── tests/                         # Unit and integration tests
│   ├── test_app.py                # Streamlit app tests
│   ├── test_augmentation.py       # Augmentation tests
│   ├── test_data_loading.py       # Data loading tests
│   ├── test_evaluation.py         # Evaluation tests
│   ├── test_inference.py          # Inference tests
│   ├── test_model.py              # Model architecture tests
│   ├── test_prediction.py         # Prediction pipeline tests
│   ├── test_preprocessing.py      # Preprocessing tests
│   └── test_training.py           # Training pipeline tests
│
├── notebooks/                     # Jupyter notebooks
│   └── eda.ipynb                  # Exploratory data analysis
│
├── docs/                          # Documentation
│   ├── architecture.md            # This file
│   ├── report.pdf                 # Project report
│   ├── presentation.pptx          # Project presentation
│   └── screenshots/               # Project screenshots
│
├── tools/                         # Utility scripts
│   └── verify_processed.py        # Verify processed dataset
│
├── app.py                         # Streamlit application (entry point)
├── README.md                      # Project README (main documentation)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── .git/                          # Git repository

```

---

## Data Pipeline

### 1. Raw Dataset

- **Source:** External vehicle image dataset
- **Location:** `data/raw/` (not included in repository)
- **Format:** Image files organized by class folders
- **Classes:** 10 vehicle types

### 2. Preprocessing (src/data/preprocess.py)

Input: Raw images
↓
- **Convert to RGB:** Handle RGBA, Grayscale, Palette images by converting to RGB with white background for transparency
- **Resize:** Standardize all images to 128×128 pixels
- **Split:** Allocate 15% for test, remaining for train/validation (80/20)
- **Normalize:** Store as JPEG (quality 95) with normalized pixel values
- **Validate:** Ensure all images match specifications

Output: Organized folder structure
```
data/processed/
├── train/          (≈6100 images)
├── validation/     (≈1400 images)
├── test/           (≈1575 images)
└── manifest.csv    (File index and metadata)
```

### 3. Data Augmentation (src/data/augmentation.py)

Applied during training only (not on test set):

| Augmentation | Enabled | Parameters |
|---|---|---|
| Rotation | Yes | ±10 degrees |
| Zoom | Yes | ±10% |
| Horizontal Flip | Yes | 50% probability |
| Vertical Flip | No | Disabled (vehicles) |
| Brightness | Yes | ±20% |

**Purpose:** Increase effective training set diversity and improve generalization

### 4. Data Loading (src/data/loader.py)

- Load configuration from YAML
- Create TensorFlow Dataset from directories
- Batch images for training (batch_size=32)
- Normalize pixel values to [0, 1] range

---

## Model Architecture

### Input

```
Shape: (128, 128, 3)  [RGB image]
```

### Convolutional Feature Extraction

Three conv→pool blocks with progressive filter expansion:

```
Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)  [→ 64×64×32]
Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)  [→ 32×32×64]
Conv2D(128, 3×3, ReLU) → MaxPooling2D(2×2) [→ 16×16×128]
```

### Regularization

```
Dropout(0.3)    [after conv layers]
Flatten         [linearize feature map]
```

### Classification Head

```
Dense(256, ReLU)     [feature transformation]
Dropout(0.4)         [regularization]
Dense(10, Softmax)   [10-class prediction]
```

### Key Specifications

| Property | Value |
|---|---|
| Total Parameters | 8,484,682 |
| Total Size | 97.1 MB |
| Input Shape | (128, 128, 3) |
| Output Classes | 10 |
| Dense Hidden Units | 256 |
| Dropout Rates | 0.3 (conv), 0.4 (dense) |
| Activation Functions | ReLU (hidden), Softmax (output) |

**Design Rationale:**
- No BatchNormalization: Keeps architecture simple and transparent
- No Transfer Learning: From-scratch CNN per project requirements
- Three conv blocks: Standard pattern for 128×128 resolution
- Dual dropout: Regularizes both spatial features and dense layers
- Dense(256): Increased capacity from baseline (128) based on Experiment 2 findings

---

## Training Pipeline

### Configuration (src/config/config_phase10_5_exp3.yaml)

| Setting | Value |
|---|---|
| Epochs | 50 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Early Stopping | Yes (patience=5) |
| Learning Rate Reduction | Yes (patience=2, factor=0.5) |

### Training Process (src/training/train.py)

1. **Initialize:** Load config, build model, compile
2. **Augment:** Apply augmentations to training batches only
3. **Train:** Run for up to 50 epochs with callbacks:
   - `EarlyStopping`: Stop if validation loss doesn't improve for 5 epochs
   - `ReduceLROnPlateau`: Reduce LR by 50% if plateau detected (patience=2)
   - `ModelCheckpoint`: Save best model checkpoint
4. **Monitor:** Track training/validation loss and accuracy
5. **Save:** Checkpoint best model and training history

### Output

- **Model:** `models/best_model_exp3.keras`
- **Metrics:** `results/metrics_exp3/training_history.json`
- **Plots:** `results/plots_exp3/{training_accuracy,training_loss}.png`

---

## Evaluation Pipeline

### Evaluation Process (src/evaluation/evaluate.py)

1. **Load:** Best model and test set
2. **Predict:** Generate predictions for all test images
3. **Compute Metrics:**
   - Test accuracy and loss
   - Per-class precision, recall, F1-score
   - Macro and weighted averages
4. **Visualize:**
   - Confusion matrix heatmap
   - Per-class performance comparison
5. **Save:** Metrics and visualizations

### Output

```
results/evaluation_exp3/
├── test_metrics.json           # Test set accuracy/loss
├── per_class_metrics.csv       # Per-class metrics
├── classification_report.csv   # Detailed breakdown
├── confusion_matrix.csv        # Confusion matrix data
└── confusion_matrix.png        # Confusion matrix heatmap
```

### Final Metrics

| Metric | Value |
|---|---|
| Test Accuracy | 72.66% |
| Test Loss | 0.7765 |
| Macro Precision | 0.674 |
| Macro Recall | 0.750 |
| Macro F1 | 0.701 |
| Weighted Precision | 0.748 |
| Weighted Recall | 0.727 |
| Weighted F1 | 0.730 |
| Best Validation Accuracy | 72.93% |
| Test Set Size | 4,075 images |

---

## Inference Pipeline

### Single-Image Prediction (src/inference/predict.py)

```
Image File
↓ Load and convert to RGB
↓ Resize to 128×128
↓ Normalize to [0, 1]
↓ Add batch dimension
↓ Forward pass through model
↓ Get softmax probabilities
↓ Return top-k predictions with confidence
```

### CLI Interface

```bash
python -m src.inference.predict \
  --config src/config/config_phase10_5_exp3.yaml \
  --image "path/to/vehicle.jpg"
```

Output:
```
Predicted Class: car
Confidence: 85.3%
Top 3 Predictions:
  1. car (85.3%)
  2. truck (8.2%)
  3. bus (4.1%)
```

### Application Interface (Streamlit - app.py)

- Upload vehicle image via web interface
- Display uploaded image preview
- Run inference and show results
- Display top-3 predictions with confidence scores
- List all supported vehicle classes

---

## Experiment History

### Baseline (config.yaml)
- Dense(128)
- 30 epochs
- Test Accuracy: ~70%

### Experiment 2 (config_phase10_5_exp2.yaml)
- **Change:** Dense(128) → Dense(256)
- Epochs: 30
- **Result:** Test Accuracy improved

### Experiment 3 (config_phase10_5_exp3.yaml) ✓ **FINAL**
- **Change:** Epochs 30 → 50
- Dense(256) maintained
- Everything else identical to Exp2
- **Result:** Test Accuracy 72.66% (Experiment 3 selected as final)

**Rationale for selecting Experiment 3:**
- Longer training budget (50 epochs vs 30) allows model to learn more
- Early stopping ensures no overfitting
- All experiment artifacts preserved for reproducibility
- Project completion requires finalizing one model; Exp3 represents best effort
- Metrics verified and documented

---

## Testing

All components have unit tests to ensure correctness:

```bash
pytest -q
# Result: 166 tests passed
```

**Test Coverage:**
- Data loading and validation
- Preprocessing and augmentation
- Model architecture and compilation
- Training pipeline
- Evaluation metrics
- Inference prediction
- Streamlit application
- Configuration loading

---

## Deployment

### Local Deployment (Streamlit)

1. Install dependencies: `pip install -r requirements.txt`
2. Run app: `streamlit run app.py`
3. Access at: http://localhost:8501

### Model Files Required
- `models/best_model_exp3.keras`
- `src/config/config_phase10_5_exp3.yaml`

### Environment Setup
- Python 3.8+
- TensorFlow 2.16.1
- All dependencies in `requirements.txt`

---

## Key Design Decisions

1. **From-Scratch CNN:** No transfer learning - learn representations specific to vehicles
2. **128×128 Resolution:** Balance between detail and computational efficiency
3. **Three Conv Blocks:** Standard pattern proven effective at this resolution
4. **Dropout Regularization:** Prevent overfitting without BatchNormalization
5. **YAML Configuration:** All hyperparameters centralized for reproducibility
6. **Modular Pipeline:** Separate concerns (data, model, training, inference) for maintainability
7. **Comprehensive Testing:** 166 tests ensure all components work correctly
8. **Experiment Tracking:** Each experiment has separate outputs for reproducibility

---

## Performance Analysis

### Strengths
- Motorcycle (F1=0.827): Distinct shape recognized well
- Bus (F1=0.846): Clear size differentiation
- Boat (F1=0.803): High recall, fewer false negatives

### Challenges
- Taxi (F1=0.517): Often confused with cars and minibuses
- Minibus (F1=0.558): Overlapping appearance with buses and cars
- Bicycle (F1=0.692): Variable appearance, small sample size

### Imbalanced Dataset
Test set has significant class imbalance:
- Car: 956 images (highest)
- Boat: 921 images
- Helicopter: 76 images (lowest)
- Taxi: 102 images

This affects macro metrics (equal class importance) vs weighted metrics (by frequency).

---

## Future Improvements

1. **Data Augmentation:** More aggressive augmentation for minority classes
2. **Class Weighting:** Inverse frequency weighting to handle imbalance
3. **Ensemble Methods:** Combine multiple models for better predictions
4. **Transfer Learning:** Fine-tune a pretrained backbone if performance insufficient
5. **Attention Mechanisms:** Help model focus on discriminative regions
6. **Data Collection:** More training data, especially for minority classes
7. **Post-Processing:** Confidence thresholding and rejection options
8. **Model Compression:** Quantization and pruning for deployment efficiency

---

## Reproducibility

### Reproduce Experiment 3

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train (if dataset available)
python -m src.training.train --config src/config/config_phase10_5_exp3.yaml

# Evaluate
python -m src.evaluation.evaluate --config src/config/config_phase10_5_exp3.yaml

# Infer
python -m src.inference.predict --config src/config/config_phase10_5_exp3.yaml --image "path/to/image.jpg"

# Test
pytest -q
```

### Key Factors for Reproducibility
- **Seed:** Set to 42 in config for deterministic preprocessing
- **Configuration:** Experiment 3 config locked in `config_phase10_5_exp3.yaml`
- **Model Checkpoint:** Saved weights in `models/best_model_exp3.keras`
- **Dependencies:** Frozen versions in `requirements.txt`
- **Evaluation Artifacts:** All metrics and plots preserved in `results/evaluation_exp3/`

---

## Contact & Attribution

**Project:** Vehicle Image Classification System
**Internship:** Bright Hub Private Limited AI Internship, Project 1
**Framework:** TensorFlow/Keras
**Status:** Experiment 3 (FINAL)
**Date:** 2026

