# Vehicle Image Classification System

AI Internship — Project 1: Image Classification System

A Convolutional Neural Network (CNN) that classifies images into four vehicle
categories: **Car, Bus, Truck, Motorcycle**. Built with TensorFlow/Keras,
evaluated with standard classification metrics, and deployed as an
interactive Streamlit app.

> **Status:** Architecture phase complete. Implementation in progress.
> See [Roadmap](#roadmap) for current phase.

---

## Overview

| | |
|---|---|
| **Task** | Multi-class image classification (4 classes) |
| **Model** | CNN built from scratch (Conv → Pool → Dropout → Dense → Softmax) |
| **Framework** | TensorFlow / Keras |
| **Deployment** | Streamlit |
| **Dataset size** | ≥ 1,000 images across 4 classes (train/val/test split) |

---

## Project Structure

```
vehicle-classification/
├── data/                  # Dataset (raw + processed) — not committed, see data/README.md
├── src/
│   ├── config/            # config.yaml — all tunable parameters
│   ├── data/               # loading, validation, preprocessing, augmentation
│   ├── models/              # CNN architecture definition
│   ├── training/             # training loop, callbacks
│   ├── evaluation/            # metrics, confusion matrix, plots
│   ├── inference/               # prediction logic (shared by app + tests)
│   └── utils/                    # logging, helper functions
├── app/                    # Streamlit application
├── models/                 # Saved trained model weights (gitignored)
├── results/                # Metrics, plots, logs
├── tests/                  # Unit tests (pytest)
├── notebooks/               # Exploratory analysis (optional)
├── docs/                    # Report, presentation, screenshots
├── requirements.txt
└── .gitignore
```

Full architecture rationale and folder-by-folder explanation: see
[`docs/architecture.md`](docs/architecture.md) *(to be added)*.

---

## Setup

```bash
git clone <repo-url>
cd vehicle-classification
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dataset is **not included in this repository**. See
[`data/README.md`](data/README.md) for download instructions and expected
folder layout.

---

## Usage

> Commands below reflect the intended final workflow and will be filled in
> as each phase is implemented.

```bash
# Preprocess raw dataset into train/val/test splits
python -m src.data.preprocess

# Train the model
python -m src.training.train

# Evaluate on the test set
python -m src.evaluation.evaluate

# Run the Streamlit app
streamlit run app/streamlit_app.py
```

---

## Roadmap

- [x] Phase 1 — Project setup (repo, structure, `.gitignore`, README)
- [ ] Phase 2 — Dataset collection
- [ ] Phase 3 — Preprocessing
- [ ] Phase 4 — Augmentation
- [ ] Phase 5 — CNN architecture
- [ ] Phase 6 — Training
- [ ] Phase 7 — Evaluation
- [ ] Phase 8 — Deployment (Streamlit)
- [ ] Phase 9 — Testing
- [ ] Phase 10 — Documentation (report, PPT, demo video)

---

## Tech Stack

Python · TensorFlow/Keras · NumPy · Pandas · Matplotlib · scikit-learn ·
Pillow/OpenCV · Streamlit

---

## License

*(add license here, e.g., MIT)*