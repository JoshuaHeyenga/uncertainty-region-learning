import os
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
from visualization import plot_results_with_decision_boundary

# === Configuration and Constants ===
UNCERTAINTY_THRESHOLD: float = CONFIG["uncertainty_threshold"]
RANDOM_STATE: int = CONFIG["random_state"]
# GAP_RATIO: float = CONFIG["gap_ratio"]
GAP_CLASS_LABEL: int = CONFIG["gap_class_label"]
METHOD: AugmentationMethod = AugmentationMethod.SMOTE
CSV_PATH: str = generate_filename(METHOD, UNCERTAINTY_THRESHOLD, base_dir="results")

augmentation_dispatch = {
    AugmentationMethod.SMOTE: augment_smote_gap_class,
    AugmentationMethod.OVERSAMPLING: augment_oversampling_gap_class,
    AugmentationMethod.SVM_SMOTE: augment_svm_smote_gap_class,
}

thresholds = [0.3, 0.35, 0.4, 0.45]
regular_ratios = np.arange(0.3, 1.6, 0.1)
extra_ratios = np.array([0.01, 0.1, 2.5, 10.0])

gap_ratios = np.sort(np.unique(np.concatenate((regular_ratios, extra_ratios))))


def main() -> None:
    """Runs the complete experimental pipeline:"""

    for threshold in thresholds:
        for gap_ratio in gap_ratios:
            print(f"Running: threshold={threshold}, gap_ratio={round(gap_ratio, 2)}")
            X_train, X_test, y_train, y_test = prepare_data()
            classifier = clean_train_classifier(X_train, y_train)
            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"[PRE-GAP] Accuracy: {acc:.4f}")

            y_train_with_gap = assign_and_log_gap_class(
                classifier, X_train, y_train, threshold
            )
            X_aug, y_aug = augment_data(X_train, y_train_with_gap, gap_ratio)

            classifier_aug = clean_train_classifier(X_aug, y_aug)
            evaluate_on_original_training_data(
                classifier_aug, X_test, y_test, X_aug, y_aug
            )

            csv_path = csv_path = generate_filename_for_gap(
                method=METHOD.value, base_dir="results"
            )

            evaluate_and_log_model(
                classifier,
                X_test,
                y_test,
                csv_path,
                METHOD.value,
                "pre",
                RANDOM_STATE,
                threshold,
                gap_ratio=gap_ratio,
            )

            orig_count = len(X_train)
            X_orig_only = X_aug[:orig_count]
            y_orig_only = y_aug[:orig_count]

            evaluate_and_log_model(
                classifier_aug,
                X_orig_only,
                y_orig_only,
                csv_path,
                METHOD.value,
                "post",
                RANDOM_STATE,
                threshold,
                gap_ratio=gap_ratio,
            )

    plot_results()


def plot_results():
    combined_df = pd.read_csv(os.path.join("results", "smote_gap_ratio_results.csv"))

    df = combined_df[
        (combined_df["class"] == 0) & (combined_df["method"].str.startswith("smote"))
    ]

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

        pre_df = combined_df[
            (combined_df["threshold"] == threshold)
            & (combined_df["class"] == 1)
            & (combined_df["stage"] == "pre")
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

    plt.suptitle("SMOTE — Class 1 Scores vs Gap Ratio", fontsize=14)
    plt.show()


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates and splits a synthetic dataset.

    Returns:
        Tuple containing training and test splits:
        (X_train, X_test, y_train, y_test)
    """

    X, y = generate_dataset()
    return split_dataset(X, y)


def assign_and_log_gap_class(classifier, X_train, y_train, threshold) -> np.ndarray:
    """
    Identifies uncertain training samples and assigns them to the gap class.

    Args:
        classifier: Trained classifier used for confidence evaluation.
        X_train: Training features.
        y_train: Original training labels.

    Returns:
        Modified training labels with low-confidence points relabeled as gap class.
    """

    y_train_with_gap, _ = assign_gap_class(
        classifier, X_train, y_train, threshold=threshold
    )

    return y_train_with_gap


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
        target_class=GAP_CLASS_LABEL,
        gap_ratio=gap_ratio,
    )


def evaluate_on_original_training_data(
    classifier, X_train_orig, y_train_with_gap, X_aug, y_aug
):
    # Identify indices of synthetic samples
    orig_count = len(X_train_orig)
    X_orig_only = X_aug[:orig_count]
    y_orig_only = y_aug[:orig_count]

    y_pred_orig = classifier.predict(X_orig_only)
    acc_orig = accuracy_score(y_orig_only, y_pred_orig)
    print(f"[POST GAP (ON ORIGINAL TEST DATA)] Accuracy: {acc_orig:.4f}")


def plot_average_f1_per_gap_ratio():
    csv_path = os.path.join("results", "smote_gap_ratio_results.csv")
    combined_df = pd.read_csv(csv_path)

    # Focus on class-level post metrics for class=1
    df = combined_df[(combined_df["stage"] == "post") & (combined_df["class"] == 1)]

    if df.empty:
        print("No data found in the CSV.")
        return

    # Group by gap_ratio and calculate mean & std for each metric
    grouped = (
        df.groupby("gap_ratio")
        .agg(
            {
                "f1": ["mean", "std"],
                "accuracy": ["mean", "std"],
                "recall": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten multi-level columns (e.g., ("f1", "mean") -> "f1_mean")
    grouped.columns = [
        "gap_ratio",
        "f1_mean",
        "f1_std",
        "accuracy_mean",
        "accuracy_std",
        "recall_mean",
        "recall_std",
    ]

    # Sort by gap ratio for proper plotting order
    grouped = grouped.sort_values(by="gap_ratio")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        grouped["gap_ratio"],
        grouped["f1_mean"],
        yerr=grouped["f1_std"],
        fmt="-o",
        capsize=4,
        label="Average F1 across thresholds",
    )

    ax.set_xlabel("Gap Ratio")
    ax.set_ylabel("Average Post Metric")
    ax.grid(True)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_average_f1_per_gap_ratio()
