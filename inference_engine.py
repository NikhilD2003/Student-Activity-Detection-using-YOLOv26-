import cv2
import csv
import time
from collections import defaultdict, deque
from ultralytics import YOLO
import numpy as np


def run_inference_streaming(
    video_path,
    output_video,
    csv_path,
    model_path,
    progress_callback=None,
    frame_callback=None,
):

    CONF_THRESH = 0.30
    IOU_THRESH = 0.45
    IMG_SIZE = 640

    TRACKER_CFG = "botsort.yaml"
    DEVICE = None

    SMOOTHING_FRAMES = 9
    MIN_TRACK_AGE = 8

    PREVIEW_SIZE = (960, 540)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "timestamp",
        "frame",
        "student_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ])

    track_history = defaultdict(lambda: deque(maxlen=SMOOTHING_FRAMES))
    track_age = defaultdict(int)

    track_to_student = {}
    next_student_id = 1

    frame_num = 0

    # 🔥 For real-time playback feel
    start_time = time.time()

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        timestamp = frame_num / fps

        if progress_callback:
            progress_callback(frame_num / total_frames)

        results = model.track(
            source=frame,
            persist=True,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            imgsz=IMG_SIZE,
            tracker=TRACKER_CFG,
            device=DEVICE,
            verbose=False,
            rect=True
        )

        annotated = frame.copy()

        if results and results[0].boxes is not None:

            for box in results[0].boxes:

                if box.id is None:
                    continue

                raw_id = int(box.id[0])

                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, xyxy)

                class_name = model.names[cls_id]

                if raw_id not in track_to_student:
                    track_to_student[raw_id] = next_student_id
                    next_student_id += 1

                student_id = track_to_student[raw_id]

                track_history[student_id].append(class_name)
                track_age[student_id] += 1

                if track_age[student_id] < MIN_TRACK_AGE:
                    continue

                smooth_class = max(
                    set(track_history[student_id]),
                    key=track_history[student_id].count,
                )

                label = f"Student {student_id}: {smooth_class}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                csv_writer.writerow([
                    round(timestamp, 3),
                    frame_num,
                    student_id,
                    smooth_class,
                    round(conf, 3),
                    x1,
                    y1,
                    x2,
                    y2,
                ])

        writer.write(annotated)

        # ✅ VIDEO-SPEED PREVIEW
        if frame_callback:

            preview = cv2.resize(annotated, PREVIEW_SIZE)
            frame_callback(preview)

            target_time = start_time + (frame_num / fps)
            sleep_time = target_time - time.time()

            if sleep_time > 0:
                time.sleep(sleep_time)

    cap.release()
    writer.release()
    csv_file.close()

    return output_video, csv_path
