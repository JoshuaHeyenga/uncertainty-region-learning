import os
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from config import CONFIG
from dataset import generate_dataset, split_dataset
from enums import AugmentationMethod
from logger import generate_filename, generate_filename_for_gap
from model import (
    assign_gap_class,
    augment_oversampling_gap_class,
    augment_smote_gap_class,
    augment_svm_smote_gap_class,
    clean_train_classifier,
    evaluate_and_log_model,
)

# === CONFIG ===
# General Data
UNCERTAINTY_THRESHOLD: float = CONFIG["uncertainty_threshold"]
RANDOM_STATES: int = CONFIG["random_states"]
FIRST_GAP_CLASS_LABEL: int = CONFIG["first_gap_class_label"]
METHOD: AugmentationMethod = AugmentationMethod.SMOTE

# Testing Range
THRESHOLDS: float = CONFIG["synth_thresholds"]
GAP_RATIOS: float = CONFIG["gap_ratios"]

# Logging
SYNTH_DIR: str = "results/synth_dataset/"
TIMESTAMP: str = datetime.now().strftime("%m.%d_%H.%M")
FILE_NAME: str = f"synth_results_{METHOD.value}_{TIMESTAMP}.csv"
FILE_PATH: str = os.path.join(SYNTH_DIR, FILE_NAME)

augmentation_dispatch = {
    AugmentationMethod.SMOTE: augment_smote_gap_class,
    AugmentationMethod.OVERSAMPLING: augment_oversampling_gap_class,
    AugmentationMethod.SVM_SMOTE: augment_svm_smote_gap_class,
}


def main() -> None:
    for seed in RANDOM_STATES:
        get_seed_performance(seed=seed)

    plot_results()


# === PERFORMANCE MEASURING ===


def get_seed_performance(seed: int) -> None:
    """
    Args:
        seed: Chosen seed for the performance measurement.

    Runs the experimental pipeline for the given seed and saves
    the results to a csv.
    """
    for threshold in THRESHOLDS:
        for gap_ratio in GAP_RATIOS:
            print(
                f"Run (seed={seed}): threshold={threshold} & gap_ratio={round(gap_ratio, 2)}"
            )

            # == Get Base Performance ==
            X_train, X_test, y_train, y_test = prepare_data()
            classifier = clean_train_classifier(X_train, y_train)

            evaluate_and_log_model(
                classifier=classifier,
                X_test=X_test,
                Y_test=y_test,
                file_path=FILE_PATH,
                method=METHOD.value,
                stage="pre",
                seed=seed,
                threshold=threshold,
                gap_ratio=gap_ratio,
            )

            # == Get Gap-Class Performance ==
            y_train_gap, _ = assign_gap_class(
                classifier=classifier,
                X=X_train,
                Y=y_train,
                threshold=threshold,
            )

            X_aug, y_aug = augment_data(X_train, y_train_gap, gap_ratio)
            classifier_aug = clean_train_classifier(X_aug, y_aug)

            evaluate_and_log_model(
                classifier=classifier_aug,
                X_test=X_test,
                Y_test=y_test,
                file_path=FILE_PATH,
                method=METHOD.value,
                stage="post",
                seed=seed,
                threshold=threshold,
                gap_ratio=gap_ratio,
            )


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates and splits a synthetic dataset.

    Returns:
        Tuple containing training and test splits:
        (X_train, X_test, y_train, y_test)
    """

    X, y = generate_dataset()
    return split_dataset(X, y)


def augment_data(X_train, y_train_with_gap, gap_ratio) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the selected augmentation method to the gap class to balance it.

    Args:
        X_train: Training features.
        y_train_with_gap: Training labels including gap class.

    Returns:
        Tuple of (X_aug, y_aug): Augmented feature and label arrays.
    """

    augment_fn = augmentation_dispatch.get(METHOD)
    if not augment_fn:
        raise ValueError(f"Unsupported augmentation method: {METHOD}")

    return augment_fn(
        X_train,
        y_train_with_gap,
        target_class=FIRST_GAP_CLASS_LABEL,
        gap_ratio=gap_ratio,
    )


# === VISUALIZATION ===


def plot_results():
    combined_df = pd.read_csv(
        "results/synth_dataset/synth_results_smote_07.09_16.46.csv"
    )

    # Average metrics across seeds
    metric_columns = ["precision", "recall", "f1", "support", "accuracy"]
    avg_df = combined_df.groupby(
        ["method", "stage", "class", "threshold", "gap_ratio"], as_index=False
    )[metric_columns].mean()

    # Filter for class 0 and smote method
    df = avg_df[(avg_df["class"] == 0) & (avg_df["method"].str.startswith("smote"))]

    thresholds_with_data = sorted(df["threshold"].dropna().unique())
    num_plots = len(thresholds_with_data)
    ncols = 2
    nrows = (num_plots + ncols - 1) // ncols  # Round up rows as needed

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
        sharey=True,
        constrained_layout=True,
    )

    axes = axes.flatten()

    for ax, threshold in zip(axes, thresholds_with_data):
        sub_df = df[df["threshold"] == threshold]
        valid_ratios = sorted(
            sub_df[sub_df["stage"] == "post"]["gap_ratio"].dropna().unique()
        )

        if not valid_ratios:
            print(f"Skipping threshold {threshold} — no valid augmented post data.")
            ax.axis("off")
            continue

        pre_df = avg_df[
            (avg_df["threshold"] == threshold)
            & (avg_df["class"] == 1)
            & (avg_df["stage"] == "pre")
        ]
        pre = pre_df.groupby("gap_ratio").mean(numeric_only=True).loc[valid_ratios]

        post = (
            sub_df[sub_df["stage"] == "post"]
            .groupby("gap_ratio")
            .mean(numeric_only=True)
            .loc[valid_ratios]
        )

        colors = {"precision": "gold", "recall": "crimson", "f1": "dodgerblue"}
        for metric in ["precision", "recall", "f1"]:
            ax.plot(
                valid_ratios,
                post[metric],
                marker="o",
                label=f"Post {metric.capitalize()}",
                color=colors[metric],
            )
            if not pre.empty:
                ax.axhline(
                    y=pre[metric].mean(),
                    linestyle="--",
                    color=colors[metric],
                    label=f"Pre {metric.capitalize()}",
                )

        # Plot accuracy
        if "accuracy" in post.columns:
            ax.plot(
                valid_ratios,
                post["accuracy"],
                marker="s",
                linestyle="-",
                color="green",
                label="Post Accuracy",
            )
        if "accuracy" in pre.columns:
            ax.axhline(
                y=pre["accuracy"].mean(),
                linestyle="--",
                color="green",
                label="Pre Accuracy",
            )

        ax.set_title(f"Threshold = {threshold}")
        ax.set_xlabel("Gap Ratio")
        ax.grid(True)
        if ax == axes[0]:
            ax.set_ylabel("Score")
            ax.legend(loc="lower left")

    plt.suptitle("SMOTE — Class 1 Scores vs Gap Ratio (Averaged)", fontsize=14)
    plt.show()


if __name__ == "__main__":
    plot_results()
