# app.py - Streamlit-safe version
import os
import streamlit as st
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import Optional

# === IMPORTANT: Prevent TensorFlow from seeing GPUs before importing it ===
# This must be set before any tensorflow import happens.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Reduce TF logging noise if TF ends up being imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ROOT = Path(__file__).parent
MODEL_PATH_KERAS_1 = ROOT / "best_tft_enhanced.keras"
MODEL_PATH_KERAS_2 = ROOT / "tft_enhanced_final (1).keras"
SCALER_PATH = ROOT / "scaler.joblib"
EVENT_MAP_PATH = ROOT / "event_type_mapping.joblib"

st.set_page_config(page_title="TFT Enhanced - Streamlit", layout="wide")

st.title("TFT Enhanced - Cloud Resource Demand Prediction")

# Utility: lazy-load tensorflow and models only when needed
@st.cache_resource
def load_keras_model(path: Path):
    """Load a Keras model lazily and cache it. Returns None if not available."""
    if not path.exists():
        st.warning(f"Keras model not found: {path.name}")
        return None
    try:
        # import tensorflow only now
        import tensorflow as tf
        # Use load_model for .keras / SavedModel / HDF5 formats
        model = tf.keras.models.load_model(str(path), compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load Keras model {path.name}: {e}")
        return None

@st.cache_resource
def load_joblib(path: Path):
    """Load joblib artifact safely and cache."""
    if not path.exists():
        st.warning(f"Artifact not found: {path.name}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load joblib artifact {path.name}: {e}")
        return None

# load artifacts (no TF import unless model exists and user requests predictions)
scaler = load_joblib(SCALER_PATH)
event_mapping = load_joblib(EVENT_MAP_PATH)

# show basic info about what was found
st.sidebar.header("Model files detected")
st.sidebar.write(f"Keras model A: {'found' if MODEL_PATH_KERAS_1.exists() else 'missing'}")
st.sidebar.write(f"Keras model B: {'found' if MODEL_PATH_KERAS_2.exists() else 'missing'}")
st.sidebar.write(f"scaler.joblib: {'found' if SCALER_PATH.exists() else 'missing'}")
st.sidebar.write(f"event_type_mapping.joblib: {'found' if EVENT_MAP_PATH.exists() else 'missing'}")

st.markdown(
    """
    **How to use**
    - Upload a CSV with the expected features or use the sample dataset.
    - If the TF models are included in the repo, click *Predict* to run the model.
    """
)

uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
sample_df: Optional[pd.DataFrame] = None
if uploaded_file is not None:
    try:
        sample_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded CSV:")
        st.dataframe(sample_df.head())
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")

if sample_df is None:
    if st.button("Load sample (random)"):
        # Create a minimal random sample (user should upload real data)
        sample_df = pd.DataFrame({
            "cpu": np.random.rand(10),
            "memory": np.random.rand(10),
            "disk": np.random.rand(10),
            "network": np.random.rand(10)
        })
        st.dataframe(sample_df)

# Predict button triggers model loading and inference
if st.button("Predict (run model)"):
    # Choose which model to use (prefer MODEL_PATH_KERAS_1 if present)
    model_path = MODEL_PATH_KERAS_1 if MODEL_PATH_KERAS_1.exists() else MODEL_PATH_KERAS_2
    if not model_path.exists():
        st.error("No Keras model found in repository. Upload model to repo or use a different branch.")
    else:
        st.info("Loading model (this may take a few seconds)...")
        model = load_keras_model(model_path)
        if model is None:
            st.error("Model could not be loaded.")
        else:
            if sample_df is None:
                st.error("No input data available. Upload CSV or load sample data.")
            else:
                try:
                    # Example preprocess - adapt to your actual feature columns
                    features = sample_df.select_dtypes(include=[np.number]).values.astype(float)
                    if scaler is not None:
                        try:
                            features = scaler.transform(features)
                        except Exception as e:
                            st.warning(f"Scaler transform failed, proceeding without scaler: {e}")
                    # reshape for model if needed (1D -> batch)
                    if len(features.shape) == 1:
                        features = features.reshape(1, -1)
                    # If model expects time-series shape, user should adapt here
                    # This basic example uses model.predict on features
                    preds = model.predict(features)
                    st.success("Prediction complete.")
                    st.write("Raw predictions (first 10):")
                    st.write(preds[:10])
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

st.markdown("---")
st.write("If your Keras models are large and were saved with a different TF version, consider re-saving them with TF 2.12 or exporting weights only.")
