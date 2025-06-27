import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

st.set_page_config(page_title="Iris Under Eyelid Overlay", layout="centered")

mode = st.radio("Choose mode:", ["Photo", "Live Stream"])

mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYELID = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYELID = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]

def load_iris_overlay(color_name, pupil  _radius_fraction=0.22, overall_opacity=0.75):
    overlay = Image.open(f"iriscolors/{color_name.lower()}.png").convert("RGBA")
    overlay_np = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGBA2BGRA)
    alpha = cv2.GaussianBlur(overlay_np[:, :, 3], (9, 9), 0)
    h, w = alpha.shape
    pupil_r = int(min(h, w) * pupil_radius_fraction)
    cv2.circle(alpha, (w//2, h//2), pupil_r, 0, -1)
    overlay_np[:, :, 3] = (alpha * overall_opacity).astype(np.uint8)
    return overlay_np

def apply_overlay_on_overlap(image, overlay_rgba, iris_pts, eyelid_pts):
    cx = int(np.mean([p[0] for p in iris_pts]))
    cy = int(np.mean([p[1] for p in iris_pts]))
    r = int(np.mean([np.linalg.norm(np.array(p)-[cx, cy]) for p in iris_pts]))
    iris_mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(iris_mask, (cx, cy), r, 255, -1)
    eyelid_mask = np.zeros_like(iris_mask)
    cv2.fillPoly(eyelid_mask, [np.array(eyelid_pts, np.int32)], 255)
    overlap_mask = cv2.bitwise_and(iris_mask, eyelid_mask)
    ol = cv2.resize(overlay_rgba, (2*r, 2*r), interpolation=cv2.INTER_AREA)
    oh, ow = ol.shape[:2]
    tx, ty = cx - ow//2, cy - oh//2
    for y in range(oh):
        for x in range(ow):
            yy, xx = ty + y, tx + x
            if 0 <= yy < image.shape[0] and 0 <= xx < image.shape[1] and overlap_mask[yy, xx]:
                alpha = ol[y, x, 3] / 255.0
                if alpha:
                    image[yy, xx] = ((1-alpha)*image[yy, xx] + alpha*ol[y, x, :3]).astype(np.uint8)
    return image

def recolor_frame(frame, overlay_rgba):
    h, w = frame.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as fm:
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return frame
        for lm in res.multi_face_landmarks:
            eyelids = [
                [(int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)) for i in L]
                for L in (LEFT_EYELID, RIGHT_EYELID)
            ]
            irises = [
                [(int(lm.landmark[i].x*w), int(lm.landmark[i].y*h)) for i in L]
                for L in (LEFT_IRIS, RIGHT_IRIS)
            ]
            for iris_pts, eyelid_pts in zip(irises, eyelids):
                frame = apply_overlay_on_overlap(frame, overlay_rgba.copy(), iris_pts, eyelid_pts)
    return frame

# UI inputs
color = st.selectbox("Overlay color", ["aquagreeniris", "bluegreeniris","Blueiris","bluishiris",
                                       "darkbluesnowflake", "Greeniris", "greyfloweriris",
                                       "lightgrey", "snakeiris", "yellowiris"])
opacity = st.slider("Opacity", 0.0, 1.0, 0.75, 0.05)
overlay_rgba = load_iris_overlay(color, overall_opacity=opacity)

# if mode == "Photo":
uploaded = st.file_uploader("Upload a face image:", type=["jpg","png"])
if uploaded:
        img = Image.open(uploaded).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        with st.spinner("Processing..."):
            out = recolor_frame(frame.copy(), overlay_rgba)
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original", use_column_width=True)
        col2.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Overlay on overlap", use_column_width=True)
        st.download_button("Download", data=cv2.imencode(".jpg", out)[1].tobytes(),
                           file_name="iris_overlap.jpg", mime="image/jpeg")

else:  # Live Stream
    mirror_video = st.checkbox("Mirror video (recommended for selfie mode)", value=True)
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if mirror_video:
                frame = cv2.flip(frame, 1)  # 1 = horizontal flip
        out = recolor_frame(frame, overlay_rgba)
        stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB")
    cap.release()
