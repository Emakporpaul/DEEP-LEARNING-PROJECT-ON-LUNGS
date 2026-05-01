import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

# Page configuration — must be first Streamlit command
st.set_page_config(
    page_title = "AI-Powered COVID-19 Pulmonary Diagnostic Assistant",
    page_icon  = "🫁",
    layout     = "wide"
)

# Constants
IMG_HEIGHT  = 224
IMG_WIDTH   = 224
CLASS_NAMES = ['COVID', 'NORMAL']

# Load model — cached so it only loads once per session
@st.cache_resource
def load_classifier():
    """
    Downloads SavedModel zip from Hugging Face and loads it.
    Uses TF SavedModel format — compatible across TF versions.
    """
    # Download zip from Hugging Face
    zip_path = hf_hub_download(
        repo_id   = "Emakporpaul/covid19-pulmonary-diagnostic",
        filename  = "mobilenet_savedmodel.zip",
        repo_type = "model"
    )

    # Extract to /tmp directory
    extract_dir = "/tmp/mobilenet_savedmodel"
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("/tmp")

    # Load from SavedModel format — version-agnostic
    model = tf.saved_model.load(extract_dir)
    return model


# Grad-CAM functions
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='out_relu'):
    """
    Generates Grad-CAM heatmap showing which lung regions
    influenced the model's prediction most strongly.
    Uses raw 0-255 pixels — model has preprocess_input baked in.
    Runs head manually to keep gradient connection intact.
    Only works with Keras model object — not SavedModel.
    """
    base_model      = model.layers[1]   # MobileNetV2 base
    conv_layer      = base_model.get_layer(last_conv_layer_name)
    base_grad_model = tf.keras.models.Model(
        inputs  = base_model.input,
        outputs = [conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_output = base_grad_model(img_array)
        x           = model.get_layer('global_average_pooling2d_1')(base_output)
        x           = model.get_layer('dense_2')(x)
        x           = model.get_layer('batch_normalization_5')(x)
        x           = model.get_layer('dropout_4')(x)
        predictions = model.get_layer('dense_3')(x)
        loss        = predictions[:, 0]

    grads        = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    heatmap     /= tf.math.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def apply_gradcam(img_array, heatmap):
    """
    Overlays Grad-CAM heatmap on original X-ray image.
    Returns original, heatmap, and blended overlay as RGB arrays.
    """
    # Resize heatmap to image dimensions
    heatmap_resized = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))

    # Apply JET colormap: blue=low attention, red=high attention
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend original + heatmap (60% image, 40% heatmap)
    img_rgb = img_array.astype(np.uint8)
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

    return img_rgb, heatmap_colored, overlay


# App Header
st.title("🫁 AI-Powered COVID-19 Pulmonary Diagnostic Assistant")
st.markdown("""
**Deep Learning-powered COVID-19 vs Normal X-Ray Classification**

This app uses a fine-tuned **MobileNetV2** model trained on 13,808 chest 
X-ray images to classify lung X-rays as COVID-19 positive or Normal.
It also generates **Grad-CAM heatmaps** to show which lung regions 
influenced the model's decision.
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
    - COVID-19 positive
    - Normal lungs
    
    **Framework:** TensorFlow 2.21
    """)

    st.divider()
    st.markdown("**Built by:** Emakpor Paul")
    st.markdown("**Project:** AI-Powered COVID-19 Pulmonary Diagnostic Assistant with Explainable Deep Learning")

# Load model
with st.spinner("Loading model from Hugging Face — this may take a moment..."):
    model = load_classifier()

st.success("Model loaded successfully!")

# File Upload
st.header("Upload a Chest X-Ray")
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image (PNG, JPG, JPEG)",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:

    # Read and resize uploaded image
    file_bytes  = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr     = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    arr         = img_resized.astype(np.float32)

    # Feed raw 0-255 pixels — preprocess_input baked inside model
    # DO NOT apply preprocess_input here — model handles it internally
    inp = np.expand_dims(arr, axis=0)

    # Run prediction + attempt Grad-CAM
    with st.spinner("Analysing X-ray..."):

        # Prediction — works with both Keras and SavedModel
        pred_prob  = float(model(inp, training=False)[0][0])
        pred_label = CLASS_NAMES[int(pred_prob > 0.5)]
        confidence = float(pred_prob) if pred_prob > 0.5 else float(1 - pred_prob)

        # Grad-CAM — attempt gracefully, fall back if SavedModel doesn't support it
        try:
            heatmap                      = make_gradcam_heatmap(inp, model)
            img_orig, heatmap_c, overlay = apply_gradcam(arr, heatmap)
        except Exception:
            # SavedModel format does not support .get_layer() — graceful fallback
            heatmap   = None
            heatmap_c = None
            overlay   = None
            img_orig  = arr.astype(np.uint8)

    # Prediction Result
    st.divider()
    st.header("Prediction Result")

    # Color code result — red for COVID, green for Normal
    if pred_label == 'COVID':
        st.error(f"**Prediction: {pred_label}** — Confidence: {confidence:.1%}")
    else:
        st.success(f"**Prediction: {pred_label}** — Confidence: {confidence:.1%}")

    # Confidence progress bar
    st.progress(confidence, text=f"Model confidence: {confidence:.1%}")

    # Grad-CAM Visualisation
    st.divider()
    st.header("Grad-CAM Explainability")

    if heatmap is not None:
        # Full Grad-CAM — available in local Keras deployment
        st.markdown("""
        The heatmap below shows which regions of the X-ray the model 
        focused on to make its decision. **Red = high attention**, 
        **Blue = low attention**.
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img_orig,
                     caption="Original X-Ray",
                     use_container_width=True)

        with col2:
            st.image(heatmap_c,
                     caption="Grad-CAM Heatmap (Red = high attention)",
                     use_container_width=True)

        with col3:
            # Caption uses same pred_label and confidence — guaranteed consistent
            st.image(overlay,
                     caption=f"Overlay — Pred: {pred_label} ({confidence:.1%})",
                     use_container_width=True)

    else:
        # Graceful fallback — SavedModel does not support layer access for Grad-CAM
        st.markdown("""
        The original X-ray is shown below. Full **Grad-CAM heatmap** 
        explainability is available in local deployment.
        """)
        st.image(img_orig,
                 caption="Original X-Ray",
                 use_container_width=True)
        st.info(
            "Grad-CAM visualization requires the full Keras model. "
            "Run the app locally with `streamlit run src/app.py` to enable it.")

    # Medical Disclaimer
    st.divider()
    st.warning("""
    **Medical Disclaimer:** This tool is for educational and research 
    purposes only. It is not intended for clinical diagnosis. Always 
    consult a qualified medical professional for medical advice.
    """)

else:
    # Placeholder when no image uploaded
    st.info("Please upload a chest X-ray image to get started.")
    st.markdown("""
    ### How it works
    1. **Upload** a chest X-ray image (PNG or JPG)
    2. The model **analyses** the image using MobileNetV2
    3. You get a **prediction** — COVID-19 or Normal
    4. A **Grad-CAM heatmap** shows which lung regions influenced the decision
    
    ### About the Model
    - Trained on 13,808 chest X-ray images
    - 95.05% accuracy on unseen test data
    - 97.82% recall — catches 97.82% of real COVID cases
    - Grad-CAM explainability for clinical transparency
    """)