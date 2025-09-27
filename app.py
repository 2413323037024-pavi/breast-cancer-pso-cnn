import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------
# Load your trained model
# ------------------------------
MODEL_PATH = "final_pso_sgd_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------
# Preprocessing function
# ------------------------------
def preprocess_image(img):
    img = img.resize((50, 50))  # Resize to match training
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🩺 Breast Cancer Detection (PSO-SGD CNN)")
st.write("Upload a histopathology image to check if it is **Benign (No Cancer)** or **Malignant (Cancer)**.")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    processed = preprocess_image(image)

    # Predict
    prediction = model.predict(processed)
    class_idx = np.argmax(prediction, axis=1)[0]

    # Labels
    labels = ["Benign (No Cancer)", "Malignant (Cancer)"]

    st.write("### 🧾 Prediction Result:")
    st.success(f"Model Prediction: **{labels[class_idx]}**")

    # Show probabilities
    st.write("### 📊 Probabilities:")
    for i, label in enumerate(labels):
        st.write(f"{label}: {prediction[0][i]*100:.2f}%")

