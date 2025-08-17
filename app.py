import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(layout="wide")

# === Sidebar Parameters ===
st.sidebar.header("Detection Settings")

mode = st.sidebar.radio(
    "Select Detection Mode",
    ["Template Matching", "Circle Detection"]
)

threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.8, 0.01)

multi_scale = st.sidebar.checkbox("Detect all sizes", value=True)
allow_rotation = st.sidebar.checkbox("Detect rotated objects", value=True)
enable_color_filter = st.sidebar.checkbox("Enable color filtering", value=False)
color_tolerance = st.sidebar.slider("Color Tolerance (±)", 0, 50, 15, disabled=not enable_color_filter)

with st.sidebar.expander("👨‍💻 Expert Mode (OpenCV settings)"):
    method_name = st.selectbox(
        "Template Matching Method",
        ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]
    )
    method_map = {
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
    }
    method = method_map[method_name]
    use_edges = st.checkbox("Use Edge-Based Matching (Canny)", value=False)

resize_width = 800
scales = [1.2, 1.0, 0.8, 0.6, 0.4] if multi_scale else [1.0]

# === Image Upload ===
st.title("🔍 Universal Object Detection & Counting")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    h, w = image_np.shape[:2]
    if max(h, w) > resize_width:
        scale_factor = resize_width / max(h, w)
        image_np = cv2.resize(image_np, (int(w * scale_factor), int(h * scale_factor)))

    st.subheader("Step 1: Draw ROI Box (Rectangle Tool)")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="#00FF00",
        background_image=Image.fromarray(image_np),
        update_streamlit=True,
        height=image_np.shape[0],
        width=image_np.shape[1],
        drawing_mode="rect",
        key="canvas",
        initial_drawing={"version": "4.4.0", "objects": []}  # ensures clearing previous ROI
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        # always take only the latest ROI and discard previous ones
        canvas_result.json_data["objects"] = [canvas_result.json_data["objects"][-1]]

        rect = canvas_result.json_data["objects"][-1]
        x, y = int(rect["left"]), int(rect["top"])
        w, h = int(rect["width"]), int(rect["height"])
        template = image_np[y:y+h, x:x+w]

        if template.size == 0:
            st.warning("⚠️ Invalid ROI selected.")
        else:
            st.image(template, caption="Selected ROI Template", width=200)
            st.subheader("Step 2: Object Detection")

            # === Detection functions ===
            def detect_template(img, template):
                result_img = img.copy()
                found_rects = []
                angles = [0] if not allow_rotation else list(range(0, 360, 15))

                selected_hsv = None
                if enable_color_filter:
                    avg_color = cv2.mean(template)[:3]
                    selected_hsv = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_RGB2HSV)[0][0]

                for scale in scales:
                    resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
                    for angle in angles:
                        center = (resized_template.shape[1] // 2, resized_template.shape[0] // 2)
                        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(resized_template, rot_mat, (resized_template.shape[1], resized_template.shape[0]))

                        if use_edges:
                            img_proc = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 50, 150)
                            rotated_proc = cv2.Canny(cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY), 50, 150)
                        else:
                            img_proc = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                            rotated_proc = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)

                        res = cv2.matchTemplate(img_proc, rotated_proc, method)
                        loc = np.where(res >= threshold) if method != cv2.TM_SQDIFF_NORMED else np.where(res <= (1 - threshold))

                        for pt in zip(*loc[::-1]):
                            region = img[pt[1]:pt[1]+rotated.shape[0], pt[0]:pt[0]+rotated.shape[1]]
                            if region.shape[:2] != rotated.shape[:2]:
                                continue
                            if selected_hsv is not None:
                                cy, cx = region.shape[0]//2, region.shape[1]//2
                                region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
                                center_hsv = region_hsv[cy, cx]
                                h_diff = min(abs(int(center_hsv[0]) - int(selected_hsv[0])), 180 - abs(int(center_hsv[0]) - int(selected_hsv[0])))
                                if h_diff > color_tolerance:
                                    continue
                            found_rects.append([pt[0], pt[1], rotated.shape[1], rotated.shape[0]])

                final_boxes, _ = cv2.groupRectangles(found_rects, 1, 0.5)
                for i, (x, y, rw, rh) in enumerate(final_boxes):
                    cv2.rectangle(result_img, (x, y), (x+rw, y+rh), (0,255,0), 2)
                    cv2.putText(result_img, f"{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                return result_img, len(final_boxes)

            def detect_circles(img, template):
                result_img = img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                roi_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
                roi_radius = int(min(template.shape[:2]) / 2)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=roi_radius,
                                           param1=50, param2=30,
                                           minRadius=int(roi_radius*0.8), maxRadius=int(roi_radius*1.2))
                count = 0
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i, (x, y, r) in enumerate(circles[0, :]):
                        cv2.circle(result_img, (x, y), r, (0, 255, 0), 2)
                        cv2.putText(result_img, f"{i+1}", (x-r, y-r), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        count += 1
                return result_img, count

            # === Mode Selection ===
            if mode == "Template Matching":
                result, count = detect_template(image_np, template)
            elif mode == "Circle Detection":
                result, count = detect_circles(image_np, template)

            st.success(f"✅ Total Objects Detected: {count}")
            st.image(result, caption="Detection Result", width=800)
    else:
        st.info("✏️ Draw a rectangle ROI on the image above.")
