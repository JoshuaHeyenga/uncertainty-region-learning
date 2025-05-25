import os
import time

import pandas as pd


def generate_filename(method, threshold, base_dir="results"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{method}_threshold_{threshold}.csv"
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)


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
):
    row = pd.DataFrame(
        [
            {
                "method": method,
                "stage": stage,
                "seed": seed,
                "class": class_label,
                "threshold": threshold,
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
