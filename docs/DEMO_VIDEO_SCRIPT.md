# Demo Video Script — Vehicle Image Classification System

**Duration:** 3-4 minutes
**Target Audience:** Internship evaluators, potential users, portfolio viewers

---

## Scene 1: Introduction & Problem Statement (30 seconds)

**Audio:**
"Welcome to the Vehicle Image Classification System. This project demonstrates a production-ready deep learning solution for automatically classifying vehicle images into 10 categories.

The challenge: Given a photograph of a vehicle, can a neural network accurately identify whether it's a car, bus, truck, bicycle, or one of seven other vehicle types?

The answer is yes—with 72.66% accuracy on the test set."

**Visual:**
- Show project title on screen
- Display 3-4 sample images (car, bus, train, helicopter)
- Show the 10 vehicle classes in a grid layout
- Quick shot of test accuracy metric (72.66%)

---

## Scene 2: Dataset & Classes (20 seconds)

**Audio:**
"Our model was trained on over 8,600 vehicle images organized into 10 classes. The dataset is split into training, validation, and test sets to ensure fair evaluation."

**Visual:**
- Show `data/processed/` folder structure:
  - train/ (6,100 images)
  - validation/ (1,400 images)
  - test/ (1,575 images)
- Display all 10 vehicle classes as labeled images:
  1. Bicycle
  2. Boat
  3. Bus
  4. Car
  5. Helicopter
  6. Minibus
  7. Motorcycle
  8. Taxi
  9. Train
  10. Truck
- Show class distribution chart (Car and Boat have most samples)

---

## Scene 3: Project Architecture (45 seconds)

**Audio:**
"The project uses a clean, modular architecture with five main components:

First, data processing: images are preprocessed, resized to 128x128 pixels, normalized, and augmented with rotation, zoom, and brightness variations.

Second, the CNN model: a from-scratch convolutional neural network with three convolutional blocks, each followed by max pooling and dropout.

Third, training: we use Adam optimizer with early stopping and learning rate reduction to train for up to 50 epochs.

Fourth, evaluation: comprehensive metrics including confusion matrix, per-class performance, and training curves.

Fifth, inference: a web interface and command-line tool for making predictions on new images."

**Visual:**
- Show project directory tree (key folders highlighted):
  - src/data/
  - src/models/
  - src/training/
  - src/evaluation/
  - src/inference/
- Display pipeline diagram:
  Image → Preprocessing → Augmentation → CNN → Prediction → Evaluation
- Show folder structure: docs/, models/, results/

---

## Scene 4: Model Architecture (45 seconds)

**Audio:**
"The model consists of:

A 128×128×3 RGB image input, followed by three convolutional blocks. Each block has a convolutional layer with ReLU activation, followed by max pooling.

After the convolutional layers, we apply dropout to regularize, then flatten the features.

The classification head has a dense layer with 256 units and ReLU activation, another dropout layer, and finally a softmax layer with 10 output units for our 10 classes.

The model has 8.48 million trainable parameters, and we train it with categorical crossentropy loss and Adam optimization."

**Visual:**
- Show model architecture diagram:
  ```
  Input (128×128×3)
  → Conv2D(32) + Pool → Conv2D(64) + Pool → Conv2D(128) + Pool
  → Dropout(0.3) → Flatten → Dense(256) → Dropout(0.4) → Dense(10, Softmax)
  ```
- Display model summary from Keras with layer dimensions
- Show parameter counts and model size (97.1 MB)
- Highlight: 256-unit dense layer in different color

---

## Scene 5: Training Results (50 seconds)

**Audio:**
"The model was trained for 50 epochs with early stopping. Here we see the training accuracy curve increasing from about 40% to peak at 73%, and the validation accuracy follows a similar trend, reaching 72.93%.

The training loss decreases consistently from 2.0 to about 0.76, indicating the model is learning effectively.

We tested three different configurations:
- Baseline with 128 dense units achieved about 70% accuracy
- Experiment 2 increased the dense layer to 256 units, improving accuracy
- Experiment 3 extended training to 50 epochs, achieving our best result of 72.66%

All experiment artifacts are preserved for reproducibility and evaluation."

**Visual:**
- Show training_accuracy.png graph (curve increasing to ~73%)
- Show training_loss.png graph (curve decreasing to ~0.76)
- Display training configuration table:
  | Parameter | Value |
  | Epochs | 50 |
  | Batch Size | 32 |
  | Learning Rate | 0.001 |
  | Early Stopping | Yes (patience=5) |
  | LR Reduction | Yes (patience=2) |
- Show final_model.json metadata:
  - Status: final
  - Test Accuracy: 72.66%
  - Model path: models/best_model_exp3.keras

---

## Scene 6: Performance Analysis (50 seconds)

**Audio:**
"Let's look at the per-class performance. The model performs best on buses and motorcycles, achieving F1-scores above 0.82, thanks to their distinctive shapes.

Boats and trains also perform well at 0.80 and 0.72 respectively.

However, we see challenges with taxis and minibuses. Taxis often get confused with cars because they have similar shapes, achieving only 0.52 F1-score. Minibuses overlap with buses in appearance.

Overall weighted F1-score across all classes is 0.73, which balances precision and recall.

The confusion matrix visualization shows these patterns clearly—you can see the strong diagonal where predictions are correct, and the off-diagonal patterns revealing common confusions."

**Visual:**
- Show per_class_metrics.csv as a table with colors:
  - Green: High F1 (Bus 0.846, Motorcycle 0.827, Boat 0.803)
  - Yellow: Medium F1 (Helicopter 0.788, Train 0.716, Bicycle 0.692, Car 0.674)
  - Red: Low F1 (Truck 0.589, Minibus 0.558, Taxi 0.517)
- Display confusion_matrix.png heatmap
- Highlight diagonal (correct predictions) vs off-diagonal (errors)
- Show top-3 confusion pairs:
  - Taxi confused with Car
  - Minibus confused with Bus
  - Truck confused with Bus

---

## Scene 7: Live Prediction Demo (60 seconds)

**Audio:**
"Now let's see the model in action. I'll upload a test image using the web interface.

As you can see, the Streamlit application provides a simple, intuitive interface. I click 'Upload a vehicle image', select a JPG or PNG file, and the application immediately shows a preview.

When I click predict or the model processes automatically, it shows:
- The uploaded image
- The predicted vehicle class
- A confidence score
- The top 3 predictions with their probabilities"

**Visual:**
- Show Streamlit app interface in browser
- Click "Upload a vehicle image"
- Select and upload a test image (e.g., test_car.jpg)
- Show image preview displayed
- Show prediction results:
  - "Predicted Vehicle: Car"
  - "Confidence: 87.4%"
  - Top 3 predictions:
    1. Car — 87.4%
    2. Truck — 8.2%
    3. Minibus — 2.1%
- Progress bars for confidence
- Show list of all supported classes in expander
- Upload a second test image (bus or motorcycle) to show different prediction
- Display another prediction result

---

## Scene 8: Technical Infrastructure (40 seconds)

**Audio:**
"Behind the scenes, this project has comprehensive testing infrastructure. All 166 tests pass, ensuring data loading, preprocessing, augmentation, model architecture, training, evaluation, inference, and the web application all work correctly.

The project includes deployment documentation for:
- Local Streamlit deployment
- Cloud deployment to Streamlit Cloud
- Docker containerization for any cloud platform
- Model file management with Git LFS

The codebase is well-organized with clear separation of concerns:
- Data processing pipeline
- Model architecture
- Training logic
- Evaluation metrics
- Inference serving
- Web interface"

**Visual:**
- Show pytest output: "166 passed in X.XXs"
- Display folder structure highlighting modular organization:
  src/data/, src/models/, src/training/, src/evaluation/, src/inference/
- Show docs/architecture.md and docs/DEPLOYMENT.md in editor
- Show test file examples: test_model.py, test_inference.py, test_app.py
- Show .gitignore protecting large files
- Show requirements.txt with all dependencies

---

## Scene 9: Command-Line Interface (30 seconds)

**Audio:**
"In addition to the web interface, the model can also be used from the command line for batch processing or integration into other systems.

The inference module provides a simple API for making predictions programmatically or via command line."

**Visual:**
- Show terminal with command:
  ```
  python -m src.inference.predict \
    --config src/config/config_phase10_5_exp3.yaml \
    --image "test_image.jpg"
  ```
- Show output:
  ```
  Predicted Class: motorcycle
  Confidence: 92.1%
  
  Top 3 Predictions:
    1. motorcycle (92.1%)
    2. bicycle (4.2%)
    3. car (1.8%)
  ```
- Show help output: `python -m src.inference.predict --help`

---

## Scene 10: Conclusion & Future Work (20 seconds)

**Audio:**
"The Vehicle Image Classification System demonstrates a complete, production-ready deep learning pipeline. With 72.66% accuracy, comprehensive evaluation, automated testing, and multiple deployment options, this is a solid foundation for vehicle classification tasks.

Future improvements could include:
- Transfer learning from pretrained models
- Data augmentation for minority classes
- Ensemble methods combining multiple models
- Deployment to cloud platforms like AWS or Google Cloud
- Model optimization for edge devices

Thank you for watching. The project is fully documented, tested, and ready for deployment."

**Visual:**
- Show key metrics summary:
  ✓ 72.66% Test Accuracy
  ✓ 10 Vehicle Classes
  ✓ 166 Tests Passing
  ✓ Deployed Web App
  ✓ Production Ready
- Show final_model.json with "status: final"
- Show GitHub repo (if applicable)
- Show links to documentation:
  - docs/architecture.md
  - docs/DEPLOYMENT.md
  - README.md
- Fade to project title screen
- Show contact info or portfolio link

---

## Technical Notes for Recording

### Software Needed
- Python IDE or terminal
- Web browser for Streamlit app
- Screen recording software (OBS Studio, QuickTime, etc.)

### Setup Before Recording
1. Run Streamlit app locally: `streamlit run app.py`
2. Prepare test images (car.jpg, bus.jpg, etc.)
3. Have terminal open with data files
4. Browser window ready with app running
5. VSCode with source code visible

### Recording Tips
- Use 1080p or higher resolution
- Record at normal speaking pace
- Highlight important lines in code
- Pause on key metrics
- Click clearly so audience can follow
- Show actual results, not fabricated output

### Editing Suggestions
- Add background music (royalty-free)
- Include title cards between sections
- Zoom in on small text
- Show project directory tree animation
- Fade between scenes
- Add captions for key metrics

### Total Duration Target
3-4 minutes (typically 180-240 seconds)

---

## Alternative: Interactive Demo Format

If presenting live instead of recording:

**Talking Points:**
1. Open project directory, show structure
2. Run tests: `pytest -q` (show 166 passed)
3. Start app: `streamlit run app.py`
4. Upload 2-3 different vehicle images
5. Show prediction accuracy
6. Open confusion matrix image
7. Discuss per-class performance
8. Show CLI inference command
9. Display architecture document
10. Answer questions about design choices

**Time Estimate:** 10-15 minutes for live demo

---

## Files to Reference During Demo

- `README.md` — Overview and quick start
- `docs/architecture.md` — Technical architecture
- `docs/DEPLOYMENT.md` — Deployment options
- `src/models/cnn_model.py` — Model architecture code
- `results/evaluation_exp3/` — Evaluation artifacts
- `results/plots_exp3/` — Training curves
- `results/final_model.json` — Model metadata
- `requirements.txt` — Dependencies
- Test files in `tests/` — Testing infrastructure

