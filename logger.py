import os
import time
from typing import Optional

import pandas as pd


def generate_filename(method: str, threshold: float, base_dir: str = "results") -> str:
    """
    Generate a timestamped CSV filename for logging experimental results.

    Args:
        method (str): The name of the augmentation method (e.g., 'smote').
        threshold (float): The uncertainty threshold used during classification.
        base_dir (str, optional): The directory where the CSV should be saved. Defaults to "results".

    Returns:
        str: Full path to the generated CSV file.
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{method}_threshold_{threshold}_{timestamp}.csv"
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)


def generate_filename_for_gap(method: str = "smote", base_dir: str = "results") -> str:
    """
    Generate a static filename for logging gap ratio experiments.

    Args:
        method (str, optional): The augmentation method used. Defaults to "smote".
        base_dir (str, optional): Directory to save the file in. Defaults to "results".

    Returns:
        str: Full path to the gap ratio results CSV file.
    """

    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "smote_gap_ratio_results.csv")


def log_metrics_to_csv(
    file_path: str,
    method: str,
    stage: str,
    seed: int,
    class_label: int,
    threshold: float,
    precision: float,
    recall: float,
    f1: float,
    support: int,
    gap_ratio: Optional[float] = None,
) -> None:
    """
    Log evaluation metrics to a CSV file. Appends the new row if the file exists, otherwise creates a new file.

    Args:
        file_path (str): Path to the CSV file where metrics will be saved.
        method (str): Augmentation method used (e.g., 'smote', 'oversampling').
        stage (str): Stage of evaluation ('pre' or 'post' augmentation).
        seed (int): Random seed used for the experiment.
        class_label (int): The class for which metrics are being logged.
        threshold (float): Uncertainty threshold used for gap class assignment.
        precision (float): Precision score for the class.
        recall (float): Recall score for the class.
        f1 (float): F1-score for the class.
        support (int): Number of true samples for the class.
        gap_ratio (Optional[float], optional): Ratio of augmentation applied to the gap class. Defaults to None.
    """

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
