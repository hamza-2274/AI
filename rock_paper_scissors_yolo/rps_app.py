import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the model
model = YOLO("game.pt")  # Make sure this path is correct!

st.title("🪨 📄 ✂️ Rock-Paper-Scissors Detector")

# Upload image
uploaded_file = st.file_uploader("Upload an image (camera or file):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    # Run YOLO model
    results = model.predict(img_array, conf=0.5)

    # Draw results
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Detection Result", use_column_width=True)

    # Show predicted class names
    labels = results[0].boxes.cls
    class_names = results[0].names
    if len(labels) > 0:
        st.success("Prediction(s):")
        for i in labels:
            st.write(f"👉 {class_names[int(i)]}")
    else:
        st.warning("No hand gesture (rock, paper, or scissor) detected.")

