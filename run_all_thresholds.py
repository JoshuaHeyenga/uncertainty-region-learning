import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from dataset import generate_dataset, split_dataset
from logger import generate_filename
from model import (
    assign_gap_class,
    augment_oversampling_gap_class,
    augment_smote_gap_class,
    augment_svm_smote_gap_class,
    clean_train_classifier,
    evaluate_and_log_model,
)
from visualization import plot_results_with_decision_boundary


def run_all_thresholds():
    seed = 42
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    method = "svm_smote"
    base_dir = "results"
    os.makedirs(base_dir, exist_ok=True)

    # Dataset & model
    X, Y = generate_dataset()
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    for threshold in thresholds:
        # Load config and dynamically update threshold
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        config["uncertainty_threshold"] = threshold

        classifier = clean_train_classifier(X_train, Y_train)

        csv_path = generate_filename(
            method=method, threshold=threshold, base_dir=base_dir
        )

        # Pre-augmentation evaluation
        evaluate_and_log_model(
            classifier, X_test, Y_test, csv_path, method, "pre", seed, threshold
        )

        # Gap assignment and augmentation
        Y_with_gap, _ = assign_gap_class(classifier, X, Y, threshold)
        if method == "smote":
            X_aug, Y_aug = augment_smote_gap_class(X, Y_with_gap, target_class=2)
        elif method == "oversampling":
            X_aug, Y_aug = augment_oversampling_gap_class(X, Y_with_gap, target_class=2)
        elif method == "svm_smote":
            X_aug, Y_aug = augment_svm_smote_gap_class(X, Y_with_gap, target_class=2)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Post-augmentation evaluation
        classifier_aug = clean_train_classifier(X_aug, Y_aug)
        evaluate_and_log_model(
            classifier_aug, X_test, Y_test, csv_path, method, "post", seed, threshold
        )

    # Merge results and plot
    combined_df = pd.concat(
        [
            pd.read_csv(os.path.join(base_dir, f))
            for f in os.listdir(base_dir)
            if f.endswith(".csv")
        ],
        ignore_index=True,
    )
    plot_threshold_metrics(combined_df)


def plot_threshold_metrics(df):
    df = df[(df["class"] == 1) & (df["method"] == "svm_smote")]
    thresholds = sorted(df["threshold"].unique())
    pre = df[df["stage"] == "pre"].groupby("threshold").mean(numeric_only=True)
    post = df[df["stage"] == "post"].groupby("threshold").mean(numeric_only=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        thresholds, post["precision"], marker="o", label="Post Precision", color="gold"
    )
    plt.plot(
        thresholds, post["recall"], marker="o", label="Post Recall", color="crimson"
    )
    plt.plot(thresholds, post["f1"], marker="o", label="Post F1", color="dodgerblue")

    plt.axhline(
        y=pre["precision"].iloc[0], linestyle="--", color="gold", label="Pre Precision"
    )
    plt.axhline(
        y=pre["recall"].iloc[0], linestyle="--", color="crimson", label="Pre Recall"
    )
    plt.axhline(y=pre["f1"].iloc[0], linestyle="--", color="dodgerblue", label="Pre F1")

    plt.title("SVM SMOTE performance on Class 1 Metrics (Pre vs Post)")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/svm_smote_thresholds_plot.png")
    plt.show()


if __name__ == "__main__":
    run_all_thresholds()
