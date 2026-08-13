"""
app.py

Phase 10 -- Streamlit deployment for the Vehicle-10 Image Classification
System (Bright Hub Private Limited AI Internship, Project 1, Step 7).

This file is a thin UI layer only. It does NOT reimplement
preprocessing, model loading, or prediction logic -- all of that is
reused directly from the existing, already-tested Phase 8 inference
module:

    src.inference.predict

specifically:
  - load_config()            (via src.data.loader, imported inside predict)
  - get_class_names()
  - load_model()
  - verify_model_matches_classes()
  - preprocess_image()  -- NOT used directly here; see note below
  - predict_image()
  - InferenceError and subclasses (for user-facing error messages)

Note on preprocessing: preprocess_image() in predict.py takes a file
PATH on disk and opens it with PIL internally. A Streamlit file
uploader gives an in-memory UploadedFile instead. To avoid duplicating
any preprocessing logic (RGB conversion, resize, normalization) here,
this app writes the uploaded bytes to a temporary file (via Python's
tempfile module, NOT into data/raw or data/processed) and calls the
existing preprocess_image() on that temp path, exactly as the CLI
does. The temp file is deleted immediately after use.

This file never:
  - retrains or modifies the model
  - writes into data/raw or data/processed
  - overwrites models/best_model.keras
  - duplicates preprocessing/prediction/top-k logic
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from src.data.loader import load_config
from src.inference.predict import (
    InferenceError,
    PredictionResult,
    get_class_names,
    load_model,
    predict_image,
    preprocess_image,
    verify_model_matches_classes,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Same default the predict.py CLI uses, so this app is consistent with
# how the rest of the project locates config.yaml.
DEFAULT_CONFIG_PATH = "src/config/config.yaml"

SUPPORTED_EXTENSIONS = ("jpg", "jpeg", "png")


# --------------------------------------------------------------------------
# Cached resource loading (model + config loaded once per session/process,
# not on every uploaded image)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_config_and_model(config_path: str):
    """
    Load config.yaml and the trained model checkpoint exactly once
    (cached across reruns/image uploads). Reuses the existing
    load_config(), get_class_names(), load_model(), and
    verify_model_matches_classes() -- no new loading logic here.

    Returns:
        (config, class_names, model)

    Raises:
        InferenceError (or subclass) if config/model loading fails --
        the caller is responsible for catching this and showing a
        user-facing message.
    """
    config = load_config(config_path)
    class_names = get_class_names(config)
    model = load_model(config)
    verify_model_matches_classes(model, class_names)
    return config, class_names, model


# --------------------------------------------------------------------------
# Helpers (small, testable, UI-independent)
# --------------------------------------------------------------------------

def is_supported_image_filename(filename: str) -> bool:
    """Return True if filename has a supported extension (jpg/jpeg/png)."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in SUPPORTED_EXTENSIONS


def run_inference_on_bytes(
    image_bytes: bytes,
    filename: str,
    config: dict,
    class_names: list,
    model,
    top_k: int = 3,
) -> PredictionResult:
    """
    Run the existing inference pipeline on in-memory image bytes.

    Writes the bytes to a temporary file (never under data/raw or
    data/processed, never overwriting the model checkpoint), calls the
    existing preprocess_image() + predict_image() from
    src.inference.predict, then deletes the temp file.

    This function contains no preprocessing or prediction logic of its
    own -- it only bridges Streamlit's in-memory upload to the
    existing path-based inference API, so it can be unit tested
    without a running Streamlit server.
    """
    suffix = Path(filename).suffix or ".jpg"
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name

        image_array = preprocess_image(tmp_path, config)
        result = predict_image(model, image_array, class_names, top_k=top_k)
        return PredictionResult(
            image_path=filename,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            probabilities=result.probabilities,
            class_names=result.class_names,
            top_k=result.top_k,
        )
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

def render_app() -> None:
    st.set_page_config(
        page_title="Vehicle Image Classification System",
        page_icon="🚗",
        layout="centered",
    )

    st.title("Vehicle Image Classification System")
    st.write(
        "This application uses a Convolutional Neural Network (CNN), "
        "trained from scratch on the Vehicle-10 dataset, to classify "
        "an uploaded image into one of the supported vehicle categories."
    )

    # Load config + model once, cached. Any failure here (missing
    # config, missing checkpoint, class-count mismatch) is shown as a
    # clean error instead of a stack trace.
    try:
        config, class_names, model = get_config_and_model(DEFAULT_CONFIG_PATH)
    except InferenceError as exc:
        st.error(f"Unable to load the trained model: {exc}")
        st.stop()
    except FileNotFoundError:
        st.error(
            f"Configuration file not found at '{DEFAULT_CONFIG_PATH}'. "
            "Please make sure you are running this app from the project root."
        )
        st.stop()

    with st.expander("Supported classes"):
        st.write(", ".join(class_names))

    uploaded_file = st.file_uploader(
        "Upload a vehicle image",
        type=list(SUPPORTED_EXTENSIONS),
    )

    if uploaded_file is None:
        st.info("Upload a JPG, JPEG, or PNG image to get a prediction.")
        return

    if not is_supported_image_filename(uploaded_file.name):
        st.error(
            "Unable to process this image. Please upload a valid JPG, "
            "JPEG, or PNG image."
        )
        return

    image_bytes = uploaded_file.getvalue()

    # Preview
    try:
        preview_image = Image.open(io.BytesIO(image_bytes))
        st.image(preview_image, caption="Uploaded image", width=700)
    except Exception:
        st.error(
            "Unable to process this image. Please upload a valid JPG, "
            "JPEG, or PNG image."
        )
        return

    with st.spinner("Running prediction..."):
        try:
            result = run_inference_on_bytes(
                image_bytes,
                uploaded_file.name,
                config,
                class_names,
                model,
                top_k=3,
            )
        except InferenceError as exc:
            st.error(
                "Unable to process this image. Please upload a valid "
                f"JPG, JPEG, or PNG image. ({exc})"
            )
            return

    st.subheader("Vehicle Classification Result")

    col1, col2 = st.columns(2)
    col1.metric("Predicted Vehicle", result.predicted_class.title())
    col2.metric("Confidence", f"{result.confidence * 100:.2f}%")

    st.write("**Top Predictions**")
    for i, pred in enumerate(result.top_k, start=1):
        st.write(f"{i}. {pred.class_name.title()} — {pred.probability * 100:.2f}%")
        st.progress(min(max(pred.probability, 0.0), 1.0))


if __name__ == "__main__":
    render_app()