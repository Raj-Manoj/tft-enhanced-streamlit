# TFT Enhanced - Streamlit (Patched)

This archive contains patched files to make the Streamlit app deployable on Streamlit Cloud.

Files included:
- requirements.txt (pinned for Streamlit Cloud)
- app.py (lazy TF import, CUDA disabled at startup, cached resources)
- README.md (this file)

## How to use
1. Unzip and copy these files into the root of your GitHub repo (overwrite existing files).
2. Push to GitHub and deploy on Streamlit Cloud.
3. If you have Keras model files or joblib artifacts, add them to the repo root as well:
   - best_tft_enhanced.keras
   - tft_enhanced_final (1).keras
   - scaler.joblib
   - event_type_mapping.joblib

## Notes
- If your Keras models were saved with a newer TensorFlow version, re-save them using tensorflow-cpu==2.12.0 to ensure compatibility.
- After updating the repo, clear Streamlit cache and restart the app on Streamlit Cloud.
