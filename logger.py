import os
import time

import pandas as pd


def generate_filename(method, threshold, base_dir="results"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{method}_threshold_{threshold}_{timestamp}.csv"
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)


def generate_filename_for_gap(method="smote", base_dir="results"):
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "smote_gap_ratio_results.csv")


def log_metrics_to_csv(
    file_path,
    method,
    stage,
    seed,
    class_label,
    threshold,
    precision,
    recall,
    f1,
    support,
    gap_ratio=None,
):
    row = pd.DataFrame(
        [
            {
                "method": method,
                "stage": stage,
                "seed": seed,
                "class": class_label,
                "threshold": threshold,
                "gap_ratio": gap_ratio,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        ]
    )

    if os.path.exists(file_path):
        row.to_csv(file_path, mode="a", header=False, index=False)
    else:
        row.to_csv(file_path, index=False)
