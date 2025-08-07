
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

st.set_page_config(layout="wide")
st.title("YOLO Object Detection App with ROI Selection")

# Sidebar options
st.sidebar.header("Detection Settings")
confidence = st.sidebar.slider("YOLO Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Canvas for ROI selection
    st.subheader("Draw ROI (select object to detect)")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=image,
        update_streamlit=True,
        height=image_np.shape[0],
        width=image_np.shape[1],
        drawing_mode="rect",
        key="canvas",
    )

    roi = None
    if canvas_result and canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            obj = objects[0]
            left = int(obj["left"])
            top = int(obj["top"])
            width = int(obj["width"])
            height = int(obj["height"])
            roi = image_np[top:top+height, left:left+width]

    if roi is not None:
        st.image(roi, caption="Selected ROI (Template)", use_container_width=True)

        # Load YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Run detection on full image using selected ROI as template
        results = model(image_np, conf=confidence)
        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="Detection Result", use_container_width=True)

        # Count detected classes
        count = len(results[0].boxes)
        st.success(f"✅ Number of detected objects: {count}")
