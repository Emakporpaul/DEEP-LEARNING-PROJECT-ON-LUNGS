import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import zipfile
import tempfile
from pathlib import Path
from huggingface_hub import hf_hub_download

# Page configuration — must be first Streamlit command
st.set_page_config(
    page_title="AI-Powered COVID-19 Pulmonary Diagnostic Assistant",
    page_icon="🫁",
    layout="wide"
)

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['covid', 'normal']

# Load model — cached so it only loads once per session
@st.cache_resource
def load_classifier():
    """
    Downloads SavedModel zip from Hugging Face and loads it.
    Uses TF SavedModel format — compatible across TF versions.
    """
    try:
        # Download zip from Hugging Face
        with st.spinner("📥 Downloading model from Hugging Face..."):
            zip_path = hf_hub_download(
                repo_id="Emakporpaul/covid19-pulmonary-diagnostic",
                filename="mobilenet_savedmodel.zip",
                repo_type="model"
            )
        
        # Extract to temporary directory
        with st.spinner("📦 Extracting model..."):
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the directory containing saved_model.pb
            saved_model_dir = None
            for root, dirs, files in os.walk(extract_dir):
                if "saved_model.pb" in files:
                    saved_model_dir = root
                    break
            
            if saved_model_dir is None:
                st.error("Could not find saved_model.pb in the downloaded zip file")
                st.stop()
        
        # Load from SavedModel format
        with st.spinner("🧠 Loading model into memory..."):
            loaded = tf.saved_model.load(saved_model_dir)
            
            # Get the prediction function
            # For SavedModel, we need to use the serving_default signature
            predict_fn = loaded.signatures['serving_default']
            
            # Test the model with a dummy input to verify it works
            dummy_input = tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)
            dummy_output = predict_fn(dummy_input)
            
        return predict_fn
    
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# App Header
st.title("🫁 AI-Powered COVID-19 Pulmonary Diagnostic Assistant")
st.markdown("""
**Deep Learning-powered COVID-19 vs Normal X-Ray Classification**

This app uses a fine-tuned **MobileNetV2** model trained on 13,808 chest 
X-ray images to classify lung X-rays as COVID-19 positive or Normal.
""")

st.divider()

# Sidebar — Model Info
with st.sidebar:
    st.header("Model Information")
    st.markdown("""
    **Architecture:** MobileNetV2 + Custom Head
    
    **Training:** Two-phase fine-tuning
    - Phase 1: Head only (frozen base)
    - Phase 2: Last 30 layers unfrozen
    
    **Test Performance:**
    | Metric | Score |
    |---|---|
    | Accuracy | 95.05% |
    | F1 Score | 96.67% |
    | Recall | 97.82% |
    | AUC-ROC | 0.9808 |
    
    **Dataset:** 13,808 chest X-rays
    - COVID-19 positive (3,616)
    - Normal lungs (10,192)
    
    **Framework:** TensorFlow 2.15
    """)

    st.divider()
    st.markdown("**Built by:** Emakpor Paul")
    st.markdown("**GitHub:** [Project Repository](https://github.com/Emakporpaul/DEEP-LEARNING-PROJECT-ON-LUNGS)")

# Load model
predict_fn = load_classifier()
st.success("✅ Model loaded successfully from Hugging Face!")

# File Upload
st.header("Upload a Chest X-Ray")
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image (PNG, JPG, JPEG)",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    # Read and resize uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    arr = img_resized.astype(np.float32)
    
    # Prepare input (raw 0-255 pixels) - MUST be float32
    inp = np.expand_dims(arr, axis=0).astype(np.float32)
    
    # Run prediction
    with st.spinner("🔍 Analyzing X-ray..."):
        # Call the SavedModel prediction function
        outputs = predict_fn(tf.convert_to_tensor(inp))
        
        # The output key might be 'dense_3', 'output_0', or 'predictions'
        # Let's find the first output tensor
        pred_prob = float(list(outputs.values())[0][0][0])
        
        pred_label = CLASS_NAMES[int(pred_prob > 0.5)]
        confidence = float(pred_prob) if pred_prob > 0.5 else float(1 - pred_prob)
    
    # Prediction Result
    st.divider()
    st.header("Prediction Result")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(img_rgb, caption="Uploaded X-Ray", use_container_width=True)
    
    with col2:
        if pred_label == 'covid':
            st.error(f"### ⚠️ COVID-19 POSITIVE")
            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(confidence, text=f"Model confidence: {confidence:.1%}")
        else:
            st.success(f"### ✅ NORMAL")
            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(confidence, text=f"Model confidence: {confidence:.1%}")
    
    # Model Interpretation
    st.divider()
    st.header("Model Interpretation")
    
    if pred_label == 'covid':
        st.markdown("""
        **Why COVID-19?**
        - The model detected patterns consistent with:
          - Ground-glass opacities
          - Bilateral infiltrates
          - Peripheral lung abnormalities
        
        These findings are characteristic of COVID-19 pneumonia in chest X-rays.
        """)
    else:
        st.markdown("""
        **Why NORMAL?**
        - The model found no significant abnormalities
        - Clear lung fields without infiltrates
        - Normal cardiac and mediastinal contours
        
        The X-ray appears to show typical healthy lung anatomy.
        """)
    
    # Medical Disclaimer
    st.divider()
    st.warning("""
    **Medical Disclaimer:** This tool is for **educational and research purposes only**. 
    It is not a substitute for professional medical diagnosis. 
    Always consult a qualified medical professional for clinical decisions.
    """)

else:
    # Placeholder when no image uploaded
    st.info("📤 **Please upload a chest X-ray image to get started.**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### How it works
        1. **Upload** a chest X-ray image (PNG or JPG)
        2. The model **analyzes** the image using MobileNetV2
        3. You get a **prediction** — COVID-19 or Normal
        4. View **confidence scores** and model interpretation
        
        ### About the Model
        - Trained on 13,808 chest X-ray images
        - 95.05% accuracy on unseen test data
        - 97.82% recall — catches 97.82% of real COVID cases
        - Two-phase fine-tuning for optimal performance
        """)
    
    with col2:
        st.markdown("""
        ### Technical Details
        - **Architecture:** MobileNetV2 (ImageNet pretrained)
        - **Input size:** 224×224 pixels
        - **Framework:** TensorFlow 2.15
        - **Deployment:** Streamlit Cloud
        
        ### Model Metrics
        | Metric | Value |
        |--------|-------|
        | Accuracy | 95.05% |
        | Recall | 97.82% |
        | Precision | 95.55% |
        | F1 Score | 96.67% |
        | AUC-ROC | 0.9808 |
        """)