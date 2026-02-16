import pandas as pd


def compute_analytics(csv_path):

    df = pd.read_csv(csv_path)

    total_students = df["student_id"].nunique()

    # OLD (kept for compatibility if needed elsewhere)
    activity_counts = (
        df.groupby("class_name")["student_id"]
        .count()
        .sort_values(ascending=False)
    )

    # ✅ Activity distribution with frame counts + student list
    activity_distribution = (
        df.groupby("class_name")
        .agg(
            frames=("frame", "count"),
            students=("student_id", lambda x: ", ".join(map(str, sorted(x.unique())))),
        )
        .reset_index()
    )

    # ✅ Timeline for frame vs activity
    timeline = (
        df.groupby(["frame", "class_name"])
        .size()
        .unstack(fill_value=0)
    )

    # =========================================================
    # ⏱ PER-STUDENT ACTIVITY DURATION (IN SECONDS)
    # =========================================================

    # Sort for correct time difference calculation
    df = df.sort_values(["student_id", "timestamp"])

    # Time difference between consecutive detections per student
    df["time_diff"] = df.groupby("student_id")["timestamp"].diff().fillna(0)

    student_activity_duration = (
        df.groupby(["student_id", "class_name"])["time_diff"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "class_name": "activity",
                "time_diff": "duration_seconds",
            }
        )
    )

    student_activity_duration["duration_seconds"] = (
        student_activity_duration["duration_seconds"].round(2)
    )

    # =========================================================

    return {
        "total_students": total_students,
        "activity_counts": activity_counts,
        "activity_distribution": activity_distribution,
        "timeline": timeline,
        "student_activity_duration": student_activity_duration,  # ✅ NEW
        "raw_df": df,
    }
