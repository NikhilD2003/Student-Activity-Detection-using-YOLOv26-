import streamlit as st
import tempfile
import os
import cv2
import subprocess
import shutil
import pandas as pd
import plotly.express as px
import av
import logging

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from inference_engine import run_inference_streaming
from analytics import compute_analytics, compute_analytics_from_df


# ============================================================
# 🔇 SUPPRESS WEBRTC WARNINGS
# ============================================================

logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)


# ============================================================
# 🌐 RTC CONFIG
# ============================================================

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}


# ============================================================
# FFMPEG
# ============================================================

def reencode_for_browser(src, dst):
    ffmpeg_bin = shutil.which("ffmpeg")

    cmd = [
        ffmpeg_bin, "-y", "-i", src,
        "-vcodec", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-acodec", "aac",
        dst,
    ]

    subprocess.run(cmd, check=True)


# ============================================================
# SESSION STATE
# ============================================================

if "live_df" not in st.session_state:
    st.session_state.live_df = pd.DataFrame(columns=[
        "timestamp","frame","student_id","class_name",
        "confidence","x1","y1","x2","y2"
    ])
    st.session_state.model = YOLO("runs/detect/weights/best.pt")


# ============================================================
# UI
# ============================================================

st.set_page_config(layout="wide")
st.title("🎓 Student Activity Detection Dashboard")

mode = st.radio("Select Input Mode", ["Upload Video", "Live Camera"])


# ============================================================
# 🔴 LIVE MODE (CLOUD SAFE)
# ============================================================

if mode == "Live Camera":

    class VideoProcessor(VideoProcessorBase):

        def __init__(self):
            self.frame_count = 0
            self.skip = 8   # 🔥 run YOLO every N frames

        def recv(self, frame):

            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            # ✅ Warmup frames (VERY IMPORTANT)
            if self.frame_count < 10:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            annotated = img

            # 🚀 Run detection every N frames
            if self.frame_count % self.skip == 0:

                results = st.session_state.model(
                    img,
                    conf=0.3,
                    verbose=False
                )

                annotated = results[0].plot()

                if results[0].boxes is not None:

                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    cls = results[0].boxes.cls.cpu().numpy().astype(int)
                    conf = results[0].boxes.conf.cpu().numpy()

                    timestamp = self.frame_count / 30

                    for i in range(len(boxes)):

                        st.session_state.live_df.loc[
                            len(st.session_state.live_df)
                        ] = [
                            timestamp,
                            self.frame_count,
                            i,
                            st.session_state.model.names[cls[i]],
                            conf[i],
                            *boxes[i]
                        ]

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")


    col1, col2 = st.columns([2, 1])

    with col1:

        ctx = webrtc_streamer(
            key="live",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=RTC_CONFIGURATION,
            async_processing=True,
        )

    if ctx.state.playing and len(st.session_state.live_df) > 0:

        analytics = compute_analytics_from_df(st.session_state.live_df)

        with col2:

            st.subheader("📊 Live Summary")

            st.metric("Total Students", analytics["total_students"])

            fig = px.bar(
                analytics["activity_distribution"],
                x="class_name",
                y="frames",
                labels={
                    "class_name": "Activity",
                    "frames": "Total Frames"
                }
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("⏱ Duration (seconds)")

            st.dataframe(
                analytics["student_activity_duration"],
                use_container_width=True
            )


# ============================================================
# 🔵 UPLOAD MODE (UNCHANGED)
# ============================================================

elif mode == "Upload Video":

    uploaded_file = st.file_uploader(
        "Upload classroom video",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_file:

        with tempfile.TemporaryDirectory() as tmpdir:

            input_path = os.path.join(tmpdir, uploaded_file.name)

            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())

            output_video = os.path.join(tmpdir, "output_raw.mp4")
            csv_file = os.path.join(tmpdir, "detections.csv")

            if st.button("▶ Run Detection"):

                progress_bar = st.progress(0)
                frame_slot = st.empty()

                def progress_cb(p):
                    progress_bar.progress(min(int(p * 100), 100))

                def frame_cb(frame):

                    preview = cv2.resize(frame, (960, 540))
                    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

                    frame_slot.image(
                        rgb,
                        channels="RGB",
                        use_container_width=True,
                    )

                with st.spinner("Running inference..."):

                    out_vid, out_csv = run_inference_streaming(
                        input_path,
                        output_video,
                        csv_file,
                        model_path="runs/detect/weights/best.pt",
                        progress_callback=progress_cb,
                        frame_callback=frame_cb,
                    )

                st.success("Inference complete!")

                browser_video = out_vid.replace(".mp4", "_browser.mp4")

                with st.spinner("Preparing video for playback..."):
                    reencode_for_browser(out_vid, browser_video)

                with open(browser_video, "rb") as f:
                    video_bytes = f.read()

                analytics = compute_analytics(out_csv)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("🎥 Annotated Video")
                    st.video(video_bytes)

                with col2:
                    st.metric(
                        "Total Students Detected",
                        analytics["total_students"],
                    )

                st.subheader("⏱ Per-Student Activity Duration (seconds)")

                st.dataframe(
                    analytics["student_activity_duration"],
                    use_container_width=True
                )
