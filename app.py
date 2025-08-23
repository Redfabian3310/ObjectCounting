import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(layout="wide")
st.write("Streamlit version:", st.__version__)
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

# Circle-specific settings
with st.sidebar.expander("⚙️ Circle Detection Settings"):
    circle_mode = st.radio("Circle Detection Method", ["HoughCircles", "BlobDetector"])

    if circle_mode == "BlobDetector":
        st.markdown("### Blob Detector Parameters")
        min_area = st.slider("Min Circle Size (area)", 50, 2000, 200)
        max_area = st.slider("Max Circle Size (area)", 500, 10000, 5000)
        min_circularity = st.slider("Min Circularity", 0.0, 1.0, 0.7, 0.05)

    if circle_mode == "HoughCircles":
        st.markdown("### HoughCircles Parameters")
        min_radius = st.slider("Min Radius (px)", 5, 200, 10)
        max_radius = st.slider("Max Radius (px)", 10, 500, 100)
        hough_dp = st.slider("dp (Resolution Ratio)", 1.0, 3.0, 1.2, 0.1)
        hough_minDist = st.slider("minDist (Min Dist Between Centers)", 5, 100, 20)
        hough_param1 = st.slider("param1 (Canny High Threshold)", 50, 200, 100)
        hough_param2 = st.slider("param2 (Accumulator Threshold)", 10, 100, 30)

# === Tips Section ===
with st.sidebar.expander("💡 Tips for Detection"):
    if mode == "Template Matching":
        st.markdown("""
        **Template Matching**  
        - Works best when objects are **similar in shape/appearance**.  
        - Enable **multi-scale** if objects appear at different sizes.  
        - Enable **rotation** if objects may be rotated.  
        - Use **edge-based matching** for strong outlines.  
        - Lower **threshold** → finds more matches (risk of false positives).  
        """)

        if enable_color_filter:
            st.markdown("""
            **Color Filtering**  
            - Helps avoid mismatches when background is noisy.  
            - **Lower tolerance** → stricter color match.  
            - Useful for detecting same-shape but different-color objects.  
            """)

    elif mode == "Circle Detection":
        if circle_mode == "HoughCircles":
            st.markdown("""
            **HoughCircles**  
            - Best for **clear, round objects** (coins, balls, buttons).  
            - Adjust **minDist** to avoid duplicate circles.  
            - Lower **param2** → detects more circles (but noisier).  
            - Use **min/max radius** to limit detection range.  
            """)
        else:
            st.markdown("""
            **Blob Detector**  
            - Best for **imperfect or filled circles** (cells, blobs, spots).  
            - Use **min/max area** to filter sizes.  
            - Increase **circularity** if too many non-circles are detected.  
            - Works well on noisy images or biological samples.  
            """)

        if enable_color_filter:
            st.markdown("""
            **Color Filtering with Circles**  
            - Only keeps circles similar in color to selected ROI.  
            - Good for detecting same-sized circles but different colors.  
            """)

resize_width = 800
scales = [1.2, 1.0, 0.8, 0.6, 0.4] if multi_scale else [1.0]

# === Image Upload ===
st.title("🔍 Universal Object Detection & Counting")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Original full-resolution image
    original = np.array(Image.open(uploaded_file).convert("RGB"))

    # Scaled-down copy for UI (preview & ROI drawing)
    h, w = original.shape[:2]
    if max(h, w) > resize_width:
        scale_factor = resize_width / max(h, w)
        display_img = cv2.resize(original, (int(w * scale_factor), int(h * scale_factor)))
    else:
        scale_factor = 1.0
        display_img = original.copy()

    # === Step 1: ROI Selection ===
    st.subheader("Step 1: (Optional) Draw ROI Box for Template / Color Filter")

    rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    background_image = Image.fromarray(rgb_img)

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=3,
        stroke_color="#00FF00",
        background_image=background_image,
        update_streamlit=True,
        height=rgb_img.shape[0],
        width=rgb_img.shape[1],
        drawing_mode="rect",
        key=f"canvas_{uploaded_file.name}"
    )

    template = None
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        rect = canvas_result.json_data["objects"][-1]
        x, y = int(rect["left"]), int(rect["top"])
        w, h = int(rect["width"]), int(rect["height"])

        # Map ROI back to original coordinates
        x, y, w, h = int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)

        template = original[y:y+h, x:x+w]
        if template.size > 0:
            st.image(template, caption="Selected ROI Template", width=200)

    # === Step 2: Detection ===
    st.subheader("Step 2: Object Detection")

    # === Detection functions ===
    def detect_template(img, template):
        result_img = img.copy()
        found_rects = []
        angles = [0] if not allow_rotation else list(range(0, 360, 15))

        selected_hsv, selected_rgb = None, None
        if enable_color_filter and template is not None:
            avg_color = cv2.mean(template)[:3]
            selected_rgb = np.array(avg_color, dtype=np.float32)
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
                        avg_region_rgb = np.mean(region.reshape(-1, 3), axis=0)
                        avg_region_hsv = cv2.cvtColor(np.uint8([[avg_region_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

                        h_diff = min(abs(int(avg_region_hsv[0]) - int(selected_hsv[0])),
                                    180 - abs(int(avg_region_hsv[0]) - int(selected_hsv[0])))

                        rgb_diff = np.abs(avg_region_rgb - selected_rgb)
                        weighted_rgb_error = np.mean(rgb_diff)

                        if h_diff > color_tolerance or weighted_rgb_error > 40:
                            continue

                    found_rects.append([pt[0], pt[1], rotated.shape[1], rotated.shape[0]])

        final_boxes, _ = cv2.groupRectangles(found_rects, 1, 0.5)
        for i, (x, y, rw, rh) in enumerate(final_boxes):
            cv2.rectangle(result_img, (x, y), (x+rw, y+rh), (0,255,0), 2)
            cv2.putText(result_img, f"{i+1}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return result_img, len(final_boxes)

    def detect_circles(img, template=None):
        result_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if circle_mode == "BlobDetector":
            params = cv2.SimpleBlobDetector_Params()
            params.filterByCircularity = True
            params.minCircularity = min_circularity

            params.filterByArea = True
            params.minArea = min_area
            params.maxArea = max_area

            params.filterByInertia = True
            params.minInertiaRatio = 0.3

            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(gray)

            count = 0
            for i, kp in enumerate(keypoints):
                x, y = int(kp.pt[0]), int(kp.pt[1])
                r = int(kp.size / 2)

                region = img[max(0,y-r):y+r, max(0,x-r):x+r]
                if region.size == 0:
                    continue

                if enable_color_filter and template is not None:
                    avg_color = np.mean(region.reshape(-1, 3), axis=0)
                    selected_rgb = np.mean(template.reshape(-1, 3), axis=0)
                    diff = np.linalg.norm(avg_color - selected_rgb)
                    if diff > 40:
                        continue

                cv2.circle(result_img, (x, y), r, (0,255,0), 2)
                cv2.putText(result_img, f"{count+1}", (x-r, y-r),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                count += 1
            return result_img, count

        else:  # HoughCircles
            gray = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=hough_dp,
                minDist=hough_minDist,
                param1=hough_param1,
                param2=hough_param2,
                minRadius=min_radius,
                maxRadius=max_radius
            )

            count = 0
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for i, (x, y, r) in enumerate(circles):
                    region = img[max(0,y-r):y+r, max(0,x-r):x+r]
                    if region.size == 0:
                        continue

                    if enable_color_filter and template is not None:
                        avg_color = np.mean(region.reshape(-1, 3), axis=0)
                        selected_rgb = np.mean(template.reshape(-1, 3), axis=0)
                        diff = np.linalg.norm(avg_color - selected_rgb)
                        if diff > 40:
                            continue

                    cv2.circle(result_img, (x, y), r, (0,255,0), 2)
                    cv2.putText(result_img, f"{count+1}", (x-r, y-r),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    count += 1
            return result_img, count

    # === Run Detection ===
    if mode == "Template Matching":
        if template is None:
            st.warning("⚠️ Please draw an ROI for template matching.")
        else:
            result, count = detect_template(original, template)
            st.success(f"✅ Total Objects Detected: {count}")
            st.image(cv2.resize(result, (display_img.shape[1], display_img.shape[0])), 
                     caption="Detection Result", width=800)

    elif mode == "Circle Detection":
        result, count = detect_circles(original, template if enable_color_filter else None)
        st.success(f"✅ Total Circles Detected: {count}")
        st.image(cv2.resize(result, (display_img.shape[1], display_img.shape[0])), 
                 caption="Circle Detection Result", width=800)
