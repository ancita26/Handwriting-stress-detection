import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from predict import predict_stress
from utils.image_processing import extract_features
from utils.multi_writer_detection import detect_writers
from utils.stress_highlight import highlight_stress
from utils.stress_meter import stress_level
from utils.writer_identifier import identify_writer
from utils.writing_stability import stability_score
from utils.stress_heatmap import generate_heatmap
from utils.personality_detector import detect_personality
from utils.history_graph import show_history_graph


# PAGE SETTINGS
st.set_page_config(
    page_title="Handwriting Stress Detection",
    layout="wide"
)


# CUSTOM CSS
st.markdown("""
<style>

.main-title{
font-size:200px;
font-weight:bold;
text-align:center;
color:#2E8B57;
}

.result-box{
padding:20px;
border-radius:12px;
background:#f2f2f2;
margin-bottom:15px;
color:black;
font-size:18px;
}

.result-box h3{
color:#333333;
}

</style>
""", unsafe_allow_html=True)


st.markdown('<p class="main-title">Handwriting Mental Health Detection System</p>', unsafe_allow_html=True)

st.write("Upload handwriting or use live camera to detect stress, emotion and personality from handwriting.")


# EMOTION DETECTOR
def detect_emotion(score, stability):

    if score < 35:
        return "Calm"

    elif score < 65:
        return "Anxiety"

    else:
        return "Anger"


# SIDEBAR
mode = st.sidebar.selectbox(
    "Select Input Mode",
    ("Upload Image","Live Camera")
)


# IMAGE MODE
if mode == "Upload Image":

    uploaded = st.file_uploader("Upload Handwriting Image")

    if uploaded:

        image = Image.open(uploaded)

        img = np.array(image)

        st.image(image, caption="Uploaded Handwriting", width=400)

        regions = detect_writers(img)

        for i,(roi,box) in enumerate(regions):

            x,y,w,h = box

            features = extract_features(roi)

            writer_id = identify_writer(features)

            prediction,score = predict_stress(features,roi)

            stability = stability_score(roi)

            level = stress_level(score)

            emotion = detect_emotion(score, stability)

            personality = detect_personality(features, score, stability)

            col1,col2 = st.columns(2)

            with col1:

                st.markdown(f"""
                <div class="result-box">
                <h3>Writer {writer_id}</h3>
                <b>Stress Score:</b> {round(score,2)} <br>
                <b>Stress Level:</b> {level} <br>
                <b>Emotion:</b> {emotion} <br>
                <b>Personality:</b> {personality} <br>
                <b>Writing Stability:</b> {round(stability,2)}
                </div>
                """, unsafe_allow_html=True)

                st.progress(int(score))

            with col2:

                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                cv2.putText(
                    img,
                    f"W{writer_id}",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

        st.subheader("Detected Writers")

        st.image(img,width=500)

        st.subheader("Stress Highlight")

        highlighted = highlight_stress(img.copy())

        st.image(highlighted,width=500)

        st.subheader("Stress Heatmap")

        heatmap = generate_heatmap(img.copy())

        st.image(heatmap,width=500)

        st.subheader("Stress History Visualization")

        if st.button("Show Stress Graph"):
            show_history_graph()

# LIVE CAMERA MODE
if mode == "Live Camera":

    st.subheader("Capture Handwriting using Phone Camera")

    img_file = st.camera_input("Take a photo of handwriting")

    if img_file is not None:
        image = Image.open(img_file)
        img = np.array(image)

        st.image(image, caption="Captured Image")

        # -------- PROCESSING --------
        features = extract_features(img)

        prediction, score = predict_stress(features)

        st.write("Stress Level:", prediction)
        st.write("Stress Score:", round(score, 2))

        # -------- STABILITY --------
        stability = stability_score(img)
        st.write("Writing Stability:", round(stability, 2))

        # -------- HEATMAP --------
        heatmap = generate_heatmap(img.copy())
        st.subheader("Stress Heatmap")
        st.image(heatmap)

        # -------- SAVE HISTORY --------
        new_data = pd.DataFrame({
            "session": [datetime.now()],
            "stress_score": [score]
        })

        new_data.to_csv("history/stress_history.csv", mode='a', header=False, index=False)