import streamlit as st
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(layout="wide")

# === Sidebar Parameters ===
st.sidebar.header("Detection Settings")
threshold = st.sidebar.slider("Match Threshold", 50, 100, 80) / 100
strict_rotation = st.sidebar.checkbox("Strict Rotation (Only 0°)", value=True)
multi_scale = st.sidebar.checkbox("Detect All Sizes", value=True)
show_template = st.sidebar.checkbox("Show ROI Template Preview", value=True)
enable_color_filter = st.sidebar.checkbox("Enable HSV Color Filter", value=False)
color_tolerance = st.sidebar.slider("Hue Tolerance (HSV)", 0, 100, 15)

resize_width = 800
scales = [1.2, 1.0, 0.8, 0.6, 0.4] if multi_scale else [1.0]

# === Image Upload ===
st.title("🔍 Object Counting App with ROI & HSV Color Filtering")

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
        key="canvas"
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        rect = canvas_result.json_data["objects"][-1]
        x, y = int(rect["left"]), int(rect["top"])
        w, h = int(rect["width"]), int(rect["height"])
        template = image_np[y:y+h, x:x+w]

        if template.size == 0:
            st.warning("⚠️ Invalid ROI selected.")
        else:
            selected_color = None

            if enable_color_filter:
                st.subheader("Step 2: Click Inside ROI to Select Color")
                click_canvas = st_canvas(
                    background_image=Image.fromarray(template),
                    update_streamlit=True,
                    height=template.shape[0],
                    width=template.shape[1],
                    drawing_mode="transform",
                    key="color_pick"
                )

                if click_canvas.json_data and len(click_canvas.json_data["objects"]) > 0:
                    click_obj = click_canvas.json_data["objects"][-1]
                    cx = int(click_obj["left"])
                    cy = int(click_obj["top"])
                    if 0 <= cx < template.shape[1] and 0 <= cy < template.shape[0]:
                        selected_color = template[cy, cx]
                        st.markdown(f"🎯 Selected Color (RGB): {tuple(selected_color)}")
                        st.color_picker("Preview", value='#%02x%02x%02x' % tuple(selected_color), key="color_preview")

            if show_template:
                st.image(template, caption="Selected ROI Template", width=200)

            st.subheader("Step 3: Object Detection")

            def rgb_to_hsv(color):
                color = np.uint8([[color]])
                hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
                return hsv[0][0]

            def hsv_distance(hsv1, hsv2):
                """Compare only the Hue component."""
                hue_diff = abs(int(hsv1[0]) - int(hsv2[0]))
                return hue_diff

            def match_objects(img, template, threshold, strict, scales, selected_color=None, color_tol=15):
                result_img = img.copy()
                found_rects = []
                angles = [0] if strict else [0, 180]

                selected_hsv = rgb_to_hsv(selected_color) if selected_color is not None else None

                for scale in scales:
                    resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
                    for angle in angles:
                        center = (resized_template.shape[1] // 2, resized_template.shape[0] // 2)
                        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(resized_template, rot_mat,
                                                 (resized_template.shape[1], resized_template.shape[0]))

                        res = cv2.matchTemplate(img, rotated, cv2.TM_CCOEFF_NORMED)
                        loc = np.where(res >= threshold)

                        for pt in zip(*loc[::-1]):
                            region = img[pt[1]:pt[1]+rotated.shape[0], pt[0]:pt[0]+rotated.shape[1]]
                            if region.shape[:2] == rotated.shape[:2]:
                                if selected_hsv is not None:
                                    region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
                                    h1, h2 = int(region_hsv.shape[0]*0.25), int(region_hsv.shape[0]*0.75)
                                    w1, w2 = int(region_hsv.shape[1]*0.25), int(region_hsv.shape[1]*0.75)
                                    center_hsv = region_hsv[h1:h2, w1:w2]
                                    avg_hsv = center_hsv.mean(axis=(0, 1))

                                    hue_diff = hsv_distance(avg_hsv, selected_hsv)

                                    # Reject grayscale or dark objects
                                    if hue_diff > color_tol or avg_hsv[1] < 30 or avg_hsv[2] < 40:
                                        continue

                                found_rects.append([pt[0], pt[1], rotated.shape[1], rotated.shape[0]])

                # Group similar rectangles
                final_boxes, _ = cv2.groupRectangles(found_rects, 1, 0.5)
                for i, (x, y, rw, rh) in enumerate(final_boxes):
                    cv2.rectangle(result_img, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
                    cv2.putText(result_img, f"{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                return result_img, len(final_boxes)

            result, count = match_objects(
                image_np, template, threshold,
                strict_rotation, scales,
                selected_color=selected_color if enable_color_filter else None,
                color_tol=color_tolerance
            )

            st.success(f"✅ Total Objects Detected: {count}")
            st.image(result, caption="Detection Result", width=800)

    else:
        st.info("✏️ Draw a rectangle ROI on the image above.")
