import streamlit as st
import tempfile
import os
import cv2
import subprocess
import shutil
import plotly.express as px
from ultralytics import YOLO

from inference_engine import run_inference_streaming
from analytics import compute_analytics


# ============================================================
# PAGE CONFIG — MUST BE FIRST STREAMLIT COMMAND
# ============================================================
st.set_page_config(layout="wide")

st.title("🎓 Student Activity Detection Dashboard")

# ============================================================
# SESSION STATE INIT (for live mode model)
# ============================================================
if "model" not in st.session_state:
    st.session_state.model = YOLO("runs/detect/weights/best.pt")


# ============================================================
# FFMPEG RE-ENCODER
# ============================================================
def reencode_for_browser(src, dst):

    ffmpeg_bin = shutil.which("ffmpeg")

    if ffmpeg_bin is None:
        raise RuntimeError("FFmpeg not found.")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", src,
        "-vcodec", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-acodec", "aac",
        dst,
    ]

    subprocess.run(cmd, check=True)


# ============================================================
# MODE SELECTION
# ============================================================
mode = st.radio(
    "Select Input Mode",
    ["Upload Video", "Live Camera"]
)

# ============================================================
# ===================== UPLOAD VIDEO MODE =====================
# ============================================================
if mode == "Upload Video":

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
                    frame_slot.image(rgb, channels="RGB")

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

                video_bytes = open(browser_video, "rb").read()
                csv_bytes = open(out_csv, "rb").read()

                analytics = compute_analytics(out_csv)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("🎥 Annotated Video")
                    st.video(video_bytes)

                    st.download_button(
                        "⬇ Download Video",
                        video_bytes,
                        file_name="output_inference_browser.mp4",
                        mime="video/mp4",
                    )

                with col2:
                    st.subheader("📊 Summary")

                    st.metric(
                        "Total Students Detected",
                        analytics["total_students"],
                    )

                    fig = px.bar(
                        analytics["activity_distribution"],
                        x="class_name",
                        y="frames",
                        hover_data=["students"],
                    )

                    st.plotly_chart(fig)

                    st.download_button(
                        "⬇ Download CSV",
                        csv_bytes,
                        file_name="detections.csv",
                        mime="text/csv",
                    )

                st.subheader("📈 Activity Timeline")

                timeline_df = analytics["timeline"].reset_index()

                timeline_long = timeline_df.melt(
                    id_vars="frame",
                    var_name="Activity",
                    value_name="Number of Students",
                )

                fig_timeline = px.line(
                    timeline_long,
                    x="frame",
                    y="Number of Students",
                    color="Activity",
                )

                st.plotly_chart(fig_timeline)

                st.subheader("🧾 Raw Detection Log")
                st.dataframe(analytics["raw_df"].head(400))

                st.subheader("⏱ Per-Student Activity Duration (seconds)")
                st.dataframe(analytics["student_activity_duration"])


# ============================================================
# ===================== LIVE CAMERA MODE ======================
# ============================================================
else:
    st.info("Live camera works only on local system (not Streamlit Cloud).")
