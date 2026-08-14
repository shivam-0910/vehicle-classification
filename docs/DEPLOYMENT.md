# Deployment Guide

This document describes how to deploy the Vehicle Image Classification System locally, to cloud platforms, and as Docker containers.

---

## Table of Contents

1. [Local Deployment (Streamlit)](#local-deployment)
2. [Cloud Deployment (Streamlit Cloud)](#streamlit-cloud)
3. [Docker Deployment](#docker)
4. [Environment Variables](#environment)
5. [Model Files & Weights](#model-files)
6. [Troubleshooting](#troubleshooting)

---

## Local Deployment

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation Steps

#### 1. Clone Repository

```bash
git clone <repository-url>
cd vehicle-classification
```

#### 2. Create Virtual Environment

```bash
# Using Python venv
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import tensorflow; import streamlit; print('✓ Dependencies OK')"
```

#### 4. Run Application

```bash
streamlit run app.py
```

#### 5. Access Application

Open your browser and navigate to:
```
http://localhost:8501
```

### Configuration

Default configuration loaded from:
```
src/config/config_phase10_5_exp3.yaml
```

To use a different config, modify `app.py` line:
```python
DEFAULT_CONFIG_PATH = "src/config/config_phase10_5_exp3.yaml"
```

### Stopping the Application

Press `Ctrl+C` in the terminal running Streamlit.

---

## Streamlit Cloud Deployment

### Prerequisites

- GitHub account
- Streamlit Cloud account (streamlit.io)
- Repository pushed to GitHub

### Deployment Steps

#### 1. Connect Repository to Streamlit Cloud

1. Go to [Streamlit Cloud](https://share.streamlit.io/)
2. Sign in with GitHub account
3. Click "New app"
4. Select your repository
5. Choose main branch
6. Set main file to: `app.py`
7. Click "Deploy"

#### 2. Configure Streamlit Cloud

In your repository, create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
maxUploadSize = 200

[logger]
level = "info"

[server]
port = 8501
maxUploadSize = 200
enableXsrfProtection = true
```

#### 3. Handle Large Model File

Since the model file is in `.gitignore`, you need to handle it:

**Option A: Git LFS (Recommended)**

```bash
# Install Git LFS
git lfs install

# Track model file
git lfs track "models/best_model_exp3.keras"

# Commit
git add models/best_model_exp3.keras .gitattributes
git commit -m "Add model with Git LFS"
git push
```

**Option B: Download from URL**

Modify `app.py` to download the model on first run:

```python
import os
import requests

def download_model_if_missing():
    model_path = "models/best_model_exp3.keras"
    if not os.path.exists(model_path):
        print("Downloading model...")
        url = "https://your-hosting-service/best_model_exp3.keras"
        response = requests.get(url)
        os.makedirs("models", exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("✓ Model downloaded")

download_model_if_missing()
```

**Option C: Secrets Management**

1. In Streamlit Cloud dashboard, go to app settings
2. Add "Secrets" section
3. Store model download URL or credentials
4. Access in app: `st.secrets["model_url"]`

#### 4. Deploy

Push changes to GitHub:
```bash
git add .streamlit/
git commit -m "Add Streamlit Cloud config"
git push
```

Streamlit Cloud will automatically redeploy.

#### 5. Access Deployed App

After deployment, your app is available at:
```
https://<your-github-username>-<repository-name>-<random-hash>.streamlitapp.com
```

---

## Docker Deployment

### Prerequisites

- Docker installed and running
- Docker Hub account (optional, for pushing images)

### Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .
COPY .streamlit/ ./.streamlit/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### .dockerignore

Create `.dockerignore`:

```
__pycache__/
*.pyc
.venv/
.git/
.gitignore
.pytest_cache/
data/raw/
data/processed/
results/
notebooks/
tests/
docs/
tools/
.vscode/
.DS_Store
*.ipynb
README.md
```

### Build Docker Image

```bash
docker build -t vehicle-classification:latest .
```

### Run Docker Container

```bash
# Basic run
docker run -p 8501:8501 vehicle-classification:latest

# With volume mount (for persistent uploads)
docker run -p 8501:8501 \
  -v ./uploads:/app/uploads \
  vehicle-classification:latest

# Run in background
docker run -d -p 8501:8501 --name vehicle-app vehicle-classification:latest
```

### Access Container

Open browser to `http://localhost:8501`

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    container_name: vehicle-classification
    ports:
      - "8501:8501"
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    environment:
      - TF_CPP_MIN_LOG_LEVEL=2
    restart: unless-stopped
```

Run with Docker Compose:

```bash
docker-compose up -d
```

### Deploy to Docker Hub

```bash
# Tag image
docker tag vehicle-classification:latest <username>/vehicle-classification:latest

# Login
docker login

# Push
docker push <username>/vehicle-classification:latest
```

---

## Environment Variables

### TensorFlow Settings

```bash
# Disable oneDNN optimizations (if having numeric issues)
export TF_ENABLE_ONEDNN_OPTS=0

# Control logging
export TF_CPP_MIN_LOG_LEVEL=2  # 0=all, 1=info, 2=warning, 3=error

# GPU configuration (if using GPU)
export CUDA_VISIBLE_DEVICES=0
```

### Streamlit Settings

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = false

[client]
showErrorDetails = true

[logger]
level = "info"
```

### Application Config

Modify config paths in `app.py`:

```python
DEFAULT_CONFIG_PATH = os.getenv(
    "MODEL_CONFIG",
    "src/config/config_phase10_5_exp3.yaml"
)
```

---

## Model Files & Weights

### Required Files for Deployment

```
models/
└── best_model_exp3.keras        (97.1 MB)

src/config/
└── config_phase10_5_exp3.yaml   (~2 KB)

src/
├── data/
├── models/
├── inference/
└── utils/
```

### Model Size Management

The final model (`best_model_exp3.keras`) is **97.1 MB**.

**For Git:**
```bash
git lfs install
git lfs track "models/*.keras"
```

**For Cloud Deployment:**
- Streamlit Cloud has upload limits (~1GB total)
- Use Git LFS for reliable handling
- Or download from cloud storage on startup

**For Docker:**
- Include model in image for faster startup
- Or download from S3/cloud storage

### Pre-trained Checkpoint Locations

```
models/best_model_exp3.keras      ← Final model (USE THIS)
models/best_model_exp2.keras      ← Experiment 2 (archived)
models/best_model_bn.keras        ← Baseline (archived)
```

Only `best_model_exp3.keras` is actively used.

---

## Performance Optimization

### Streamlit Optimization

1. **Enable caching:**
```python
@st.cache_resource
def load_model():
    return keras.models.load_model("models/best_model_exp3.keras")
```

2. **Reduce image size:** Limit `maxUploadSize` in config
3. **Batch processing:** Not needed for single-image predictions

### Model Optimization

```python
# Quantization (convert to 8-bit)
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Pruning (remove small weights)
# Quantization-Aware Training (QAT)
```

### Memory Management

```bash
# Limit GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Use mixed precision
# Set TensorFlow optimization level
export TF_DETERMINISTIC_OPS=1
```

---

## Troubleshooting

### Model Not Found

**Error:** `FileNotFoundError: models/best_model_exp3.keras`

**Solution:**
1. Verify file exists: `ls models/best_model_exp3.keras`
2. Check working directory: Application should run from project root
3. For cloud: Ensure model is committed (Git LFS) or downloaded at startup

### Out of Memory

**Error:** `ResourceExhaustedError` or system memory issues

**Solution:**
```bash
# Limit TensorFlow memory
export TF_GPU_MEMORY_FRACTION=0.5

# Or in code:
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Slow Predictions

**Causes:** Model loading on each prediction, large image, CPU inference

**Solutions:**
1. Use `@st.cache_resource` to load model once
2. Optimize image preprocessing
3. Use GPU if available
4. Consider model quantization

### Config Not Found

**Error:** `FileNotFoundError: src/config/config_phase10_5_exp3.yaml`

**Solution:**
1. Verify config file exists
2. Use absolute path: `os.path.join(os.getcwd(), "src/config/...")`
3. Check working directory in deployment

### CORS/Security Issues

**Solution:** Update `.streamlit/config.toml`
```toml
[server]
enableCORS = false
enableXsrfProtection = true
```

### TensorFlow Version Mismatch

**Error:** Model saved with different TF version

**Solution:**
```bash
# Install exact version
pip install tensorflow==2.16.1

# Or rebuild model with current version
python -c "from src.models.cnn_model import build_and_compile_model; ..."
```

---

## Deployment Checklist

Before deploying to production:

- [ ] Model file present: `models/best_model_exp3.keras`
- [ ] Config file present: `src/config/config_phase10_5_exp3.yaml`
- [ ] Tests pass: `pytest -q` (166 passed)
- [ ] Requirements frozen: `requirements.txt` with versions
- [ ] `.streamlit/config.toml` created
- [ ] Environment variables set
- [ ] Model download tested (if cloud)
- [ ] Streamlit app runs locally: `streamlit run app.py`
- [ ] Image upload/inference works
- [ ] Error handling tested
- [ ] Performance acceptable
- [ ] Documentation updated

---

## Deployment Summary

| Platform | Difficulty | Time | Hosting | Link |
|---|---|---|---|---|
| **Local** | ⭐ Easy | 5 min | Self | http://localhost:8501 |
| **Streamlit Cloud** | ⭐ Easy | 10 min | Streamlit | share.streamlit.io |
| **Docker** | ⭐⭐ Medium | 20 min | Self/Cloud | localhost:8501 |
| **AWS** | ⭐⭐⭐ Hard | 1 hour | AWS | Custom URL |
| **Google Cloud** | ⭐⭐⭐ Hard | 1 hour | GCP | Custom URL |

**Recommendation for Internship Submission:** Deploy to Streamlit Cloud for easy access and demonstration.

---

## Contact & Support

For deployment issues:
1. Check [README.md](../README.md) for basic setup
2. Review `.streamlit/config.toml` settings
3. Check logs: `streamlit logs`
4. Verify model file exists and is readable
