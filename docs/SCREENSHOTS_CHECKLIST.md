# Screenshot Checklist — Project Submission

This document lists all required screenshots for the Vehicle Image Classification project submission.

**Status:** SUBMISSION-READY
**Note:** Screenshots should be captured manually following the guide below.

---

## Screenshots Inventory

### Location
All screenshots should be saved to: `docs/screenshots/`

### Naming Convention
Use numeric prefixes with descriptive names:
- `01_project_structure.png`
- `02_model_architecture.png`
- ... (and so on)

---

## Required Screenshots (12 total)

### 1. Project Directory Structure
**File:** `01_project_directory.png`

**What to show:**
- File explorer showing project root folder
- Visible folders: `src/`, `models/`, `results/`, `tests/`, `docs/`, `data/`
- Key files: `README.md`, `requirements.txt`, `app.py`, `.gitignore`

**How to capture:**
1. Open file explorer in project root
2. Show folder tree structure
3. Ensure all key folders are visible
4. Take screenshot

**Why needed:** Demonstrates project organization

---

### 2. Model Architecture Diagram
**File:** `02_model_architecture.png`

**What to show:**
- Model summary from Keras with layer names and shapes
- Output from: `model.summary()`
- Should show:
  - Input shape: (None, 128, 128, 3)
  - Conv layers with filter counts
  - MaxPooling layers
  - Dense(256) and Dense(10) layers
  - Total parameters: 8.48M

**How to capture:**
```bash
# Run in terminal/Python
.venv\Scripts\activate
python -c "
from tensorflow import keras
model = keras.models.load_model('models/best_model_exp3.keras')
model.summary()
"
# Take screenshot of output
```

**Why needed:** Proves model architecture meets specifications

---

### 3. Training Accuracy Curve
**File:** `03_training_accuracy.png`

**What to show:**
- Graph showing training and validation accuracy over epochs
- X-axis: Epochs (0-50)
- Y-axis: Accuracy (0.0-1.0)
- Training curve increases to ~0.73
- Validation curve follows, peaks around 0.7293

**Note:** This file already exists at: `results/plots_exp3/training_accuracy.png`

**How to capture:**
1. Open file: `results/plots_exp3/training_accuracy.png`
2. Take screenshot or copy to `docs/screenshots/03_training_accuracy.png`

**Why needed:** Shows model learning progress

---

### 4. Training Loss Curve
**File:** `04_training_loss.png`

**What to show:**
- Graph showing training and validation loss over epochs
- X-axis: Epochs (0-50)
- Y-axis: Loss (0.0-3.0)
- Both curves decrease consistently
- Final loss around 0.76

**Note:** This file already exists at: `results/plots_exp3/training_loss.png`

**How to capture:**
1. Open file: `results/plots_exp3/training_loss.png`
2. Take screenshot or copy to `docs/screenshots/04_training_loss.png`

**Why needed:** Demonstrates loss reduction during training

---

### 5. Confusion Matrix Heatmap
**File:** `05_confusion_matrix.png`

**What to show:**
- 10×10 heatmap showing prediction confusion between classes
- Color intensity indicates number of predictions
- Diagonal should be bright (correct predictions)
- Off-diagonal shows confusions (e.g., taxi-car)
- Class names on x and y axes
- Title: "Confusion Matrix"

**Note:** This file already exists at: `results/evaluation_exp3/confusion_matrix.png`

**How to capture:**
1. Open file: `results/evaluation_exp3/confusion_matrix.png`
2. Take screenshot or copy to `docs/screenshots/05_confusion_matrix.png`

**Why needed:** Visual representation of model performance

---

### 6. Classification Report (Per-Class Metrics)
**File:** `06_classification_report.png`

**What to show:**
- CSV/table of per-class metrics:
  - Class name, Precision, Recall, F1-Score, Support
  - Best performers: Bus (0.846), Motorcycle (0.827)
  - Worst performers: Taxi (0.517), Minibus (0.558)
  - Macro and Weighted averages at bottom

**How to capture:**
1. Open file: `results/evaluation_exp3/per_class_metrics.csv`
2. View in editor or spreadsheet application
3. Format nicely (can use Excel, Google Sheets)
4. Take screenshot showing all classes and averages

**Why needed:** Detailed per-class performance breakdown

---

### 7. Final Test Metrics JSON
**File:** `07_test_metrics.png`

**What to show:**
- JSON file showing:
  - test_accuracy: 0.7266
  - test_loss: 0.7765
  - num_test_images: 4075
  - num_classes: 10
  - best_validation_accuracy: 0.7293

**How to capture:**
1. Open file: `results/evaluation_exp3/test_metrics.json`
2. Display in text editor
3. Format for readability (pretty-print if possible)
4. Take screenshot

**Alternative:** Display in formatted table in README

**Why needed:** Authoritative final performance numbers

---

### 8. Streamlit Web App - Upload Screen
**File:** `08_streamlit_app_upload.png`

**What to show:**
- Streamlit application interface
- Page title: "Vehicle Image Classification System"
- Description text visible
- File uploader widget with "Upload a vehicle image" label
- "Supported classes" expander section (can be expanded or closed)
- Instructions: "Upload a JPG, JPEG, or PNG image..."

**How to capture:**
1. Run: `streamlit run app.py`
2. Open browser to: `http://localhost:8501`
3. Wait for app to load
4. Take screenshot of initial interface
5. Optional: expand "Supported classes" section and take additional screenshot

**Why needed:** Shows deployment and user interface

---

### 9. Streamlit Web App - Prediction Result
**File:** `09_streamlit_app_prediction.png`

**What to show:**
- Uploaded image displayed on screen
- Predicted class shown (e.g., "Car")
- Confidence percentage (e.g., "87.4%")
- Top 3 predictions listed with probabilities
- Progress bars showing relative confidence

**How to capture:**
1. Run: `streamlit run app.py`
2. Upload a test image (car.jpg, bus.jpg, etc.)
3. Wait for prediction to complete
4. Take screenshot showing all results
5. Ensure image, class, confidence, and top predictions all visible

**Note:** File upload and prediction should happen automatically

**Why needed:** Demonstrates working inference in web interface

---

### 10. Command-Line Inference (Terminal)
**File:** `10_cli_inference.png`

**What to show:**
- Terminal/command prompt showing:
  - Command: `python -m src.inference.predict --config src/config/config_phase10_5_exp3.yaml --image "test.jpg"`
  - Output showing:
    - Predicted Class: [vehicle type]
    - Confidence: [percentage]
    - Top 3 Predictions with probabilities

**How to capture:**
1. Activate virtual environment: `.venv\Scripts\activate`
2. Run prediction command on a test image
3. Take screenshot of command and output

**Example output:**
```
Predicted Class: car
Confidence: 85.3%

Top 3 Predictions:
  1. car (85.3%)
  2. truck (8.2%)
  3. bus (4.1%)
```

**Why needed:** Shows CLI inference capability

---

### 11. GitHub Repository
**File:** `11_github_repository.png`

**What to show:**
- GitHub repository main page
- Repository name: vehicle-classification
- README.md preview with key information
- File list showing project structure
- "About" section with description

**How to capture:**
1. If repo is on GitHub, open it in browser
2. Take screenshot of main page
3. Or: Show local repo with git status

**Note:** Only if pushing to GitHub

**Why needed:** Shows public code availability

---

### 12. Model Metadata (final_model.json)
**File:** `12_final_model_metadata.png`

**What to show:**
- JSON configuration showing:
  - experiment: "Experiment 3"
  - model_path: "models/best_model_exp3.keras"
  - config_path: "src/config/config_phase10_5_exp3.yaml"
  - test_accuracy: 0.7266257405281067
  - test_loss: 0.7764738202095032
  - status: "final"

**How to capture:**
1. Open file: `results/final_model.json`
2. View in text editor
3. Take screenshot
4. Ensure all fields are visible

**Why needed:** Official designation of final model

---

## Optional Screenshots (Recommended)

### A. Data Distribution
**File:** `A_data_distribution.png`

**What to show:**
- Bar chart showing number of images per class
- Car: 956, Boat: 921, Bus: 485, etc.
- Highlights class imbalance

**How to create:**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/evaluation_exp3/per_class_metrics.csv')
df_sorted = df.sort_values('support', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(df_sorted['class'], df_sorted['support'])
plt.xlabel('Vehicle Class')
plt.ylabel('Number of Test Images')
plt.title('Test Set Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('docs/screenshots/A_data_distribution.png', dpi=150)
```

---

### B. Performance Comparison
**File:** `B_performance_metrics.png`

**What to show:**
- Radar or bar chart comparing:
  - Precision, Recall, F1-Score
  - Macro vs Weighted metrics

**How to create:**
Use matplotlib to create comparison charts

---

### C. Model Size & Efficiency
**File:** `C_model_efficiency.png`

**What to show:**
- Table showing:
  - Total Parameters: 8.48M
  - Model Size: 97.1 MB
  - Training Time: ~50 epochs
  - Inference Time: ~500ms per image

---

## Submission Checklist

**Required (12 screenshots):**
- [ ] 01_project_directory.png
- [ ] 02_model_architecture.png
- [ ] 03_training_accuracy.png
- [ ] 04_training_loss.png
- [ ] 05_confusion_matrix.png
- [ ] 06_classification_report.png
- [ ] 07_test_metrics.png
- [ ] 08_streamlit_app_upload.png
- [ ] 09_streamlit_app_prediction.png
- [ ] 10_cli_inference.png
- [ ] 11_github_repository.png (if applicable)
- [ ] 12_final_model_metadata.png

**Recommended (optional):**
- [ ] A_data_distribution.png
- [ ] B_performance_metrics.png
- [ ] C_model_efficiency.png

---

## Screenshot Quality Guidelines

**Resolution:**
- Minimum: 1280×720 (720p)
- Recommended: 1920×1080 (1080p)
- Maximum: Reasonable file size (< 2MB each)

**Clarity:**
- Text should be clearly readable
- Use 14-16pt fonts minimum
- High contrast (dark text on light background)

**Framing:**
- Include relevant context
- Remove sensitive information (usernames, paths)
- Remove browser toolbars if not relevant

**Naming:**
- Use lowercase with underscores
- Use descriptive names (not "screenshot_1", "img_2")
- Keep consistent numbering

---

## Creating Screenshots Programmatically

### Python Approach (using Selenium)

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Start browser
driver = webdriver.Chrome()

# Navigate to local Streamlit app
driver.get("http://localhost:8501")
time.sleep(3)

# Take screenshot
driver.save_screenshot("docs/screenshots/08_streamlit_app.png")

# Close browser
driver.quit()
```

### Subprocess Approach (Terminal)

```bash
# Capture terminal output to file
python -m src.inference.predict \
  --config src/config/config_phase10_5_exp3.yaml \
  --image "test.jpg" > temp_output.txt

# Then screenshot or copy
```

---

## Organizing Screenshots in Documentation

### In README.md

```markdown
## Model Performance

### Training Progress
![Training Accuracy](docs/screenshots/03_training_accuracy.png)
![Training Loss](docs/screenshots/04_training_loss.png)

### Evaluation Results
![Confusion Matrix](docs/screenshots/05_confusion_matrix.png)

### Live Prediction
![Web App Prediction](docs/screenshots/09_streamlit_app_prediction.png)
```

### In Markdown Reports

```markdown
# Project Implementation

## Architecture
![Model Architecture](../docs/screenshots/02_model_architecture.png)

## Results
![Confusion Matrix](../docs/screenshots/05_confusion_matrix.png)
```

---

## References

- **Model output:** `results/evaluation_exp3/`
- **Training plots:** `results/plots_exp3/`
- **Application:** `app.py`
- **Metadata:** `results/final_model.json`

---

## Notes for Submission

1. **All screenshots must be real:** No doctored or fake images
2. **Include dates/times if visible:** Shows when testing was done
3. **Consistent quality:** All screenshots should have similar quality
4. **Organized folder:** All files in `docs/screenshots/`
5. **Clear filenames:** Easy to identify each screenshot's purpose
6. **Updated README:** Link screenshots where relevant

---

## Troubleshooting Screenshot Capture

**Issue:** Can't capture Streamlit app
**Solution:** Ensure app is running on localhost:8501

**Issue:** Image quality is poor
**Solution:** Use higher screen resolution, zoom in on relevant area

**Issue:** Text is too small to read
**Solution:** Increase font size in IDE/terminal, take zoomed screenshot

**Issue:** Screenshot too large
**Solution:** Compress using image optimizer or crop to relevant area

---

## Completion Status

- **Screenshots Ready:** All files exist or can be generated
- **Documentation:** Complete
- **Checklist:** Above
- **Quality:** Professional, clear, informative
- **Organization:** Numbered and named consistently

**To complete:** Capture or copy screenshots to `docs/screenshots/` folder

