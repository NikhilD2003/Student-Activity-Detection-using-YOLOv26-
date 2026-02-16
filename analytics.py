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

    # ✅ NEW — proper activity distribution table
    activity_distribution = (
        df.groupby("class_name")
        .agg(
            frames=("frame", "count"),
            students=("student_id", lambda x: ", ".join(map(str, sorted(x.unique())))),
        )
        .reset_index()
    )

    timeline = (
        df.groupby(["frame", "class_name"])
        .size()
        .unstack(fill_value=0)
    )

    return {
        "total_students": total_students,
        "activity_counts": activity_counts,
        "activity_distribution": activity_distribution,   # ✅ NEW
        "timeline": timeline,
        "raw_df": df,
    }
