import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import generate_dataset, split_dataset
from logger import generate_filename_for_gap
from model import (
    assign_gap_class,
    augment_smote_gap_class,
    clean_train_classifier,
    evaluate_and_log_model,
)


def run_gap_size_experiments():
    seed = 42
    thresholds = [0.3, 0.35, 0.4, 0.45]
    gap_ratios = np.arange(0.3, 1.6, 0.1)  # 30% to 150%
    method = "smote"  # IMPORTANT: must match filtering at bottom!
    base_dir = "results"
    os.makedirs(base_dir, exist_ok=True)

    # Load and split dataset once
    X, Y = generate_dataset()
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    for threshold in thresholds:
        for ratio in gap_ratios:
            print(f"Running: threshold={threshold}, gap_ratio={round(ratio, 2)}")

            classifier = clean_train_classifier(X_train, Y_train)
            Y_with_gap, _ = assign_gap_class(classifier, X, Y, threshold)

            X_aug, Y_aug = augment_smote_gap_class(
                X, Y_with_gap, target_class=2, gap_ratio=ratio
            )
            classifier_aug = clean_train_classifier(X_aug, Y_aug)

            csv_path = generate_filename_for_gap(method=method, base_dir=base_dir)

            evaluate_and_log_model(
                classifier,
                X_test,
                Y_test,
                csv_path,
                method,
                "pre",
                seed,
                threshold,
                gap_ratio=ratio,
            )
            evaluate_and_log_model(
                classifier_aug,
                X_test,
                Y_test,
                csv_path,
                method,
                "post",
                seed,
                threshold,
                gap_ratio=ratio,
            )

    # === Visualization: Only show subplots for thresholds with post data ===
    combined_df = pd.read_csv(os.path.join(base_dir, "smote_gap_ratio_results.csv"))

    df = combined_df[
        (combined_df["class"] == 1) & (combined_df["method"].str.startswith("smote"))
    ]

    thresholds_with_data = sorted(df["threshold"].dropna().unique())
    num_cols = len(thresholds_with_data)
    fig, axes = plt.subplots(
        1, num_cols, figsize=(4 * num_cols, 5), sharey=True, constrained_layout=True
    )

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
                    color="gray",
                    label=f"Pre {metric.capitalize()}",
                )

        ax.set_title(f"Threshold = {threshold}")
        ax.set_xlabel("Gap Ratio")
        ax.grid(True)
        if ax == axes[0]:
            ax.set_ylabel("Score")
            ax.legend(loc="upper left")

    plt.suptitle(
        "SMOTE — Class 1 Scores vs Gap Ratio",
        fontsize=14,
    )
    plt.show()


if __name__ == "__main__":
    run_gap_size_experiments()
