import matplotlib.pyplot as plt
import numpy as np
import yaml

from dataset import generate_dataset, split_dataset
from logger import generate_filename, log_metrics_to_csv
from model import (
    assign_gap_class,
    augment_oversampling_gap_class,
    augment_smote_gap_class,
    augment_svm_smote_gap_class,
    clean_train_classifier,
    evaluate_and_log_model,
)
from visualization import plot_results_with_decision_boundary

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def main():
    """
    Executes the full classification workflow including:

    1. Generating a synthetic dataset using Gaussian blobs.
    2. Training a baseline classifier on the original data.
    3. Evaluating and visualizing the decision boundary of the original model.
    4. Identifying uncertain data points using prediction confidence and assigning them to a 'gap' class (label 2).
    5. Augmenting the gap class using SMOTE to synthetically generate more representative samples.
    6. Retraining the classifier on the augmented dataset.
    7. Evaluating and visualizing the decision boundary after augmentation.

    Visual comparison of the decision boundary before and after gap-class augmentation is shown side-by-side.
    """

    seed = 42
    threshold = config["uncertainty_threshold"]
    method = "smote"  # Options: "smote", "oversampling", "svm_smote"

    X, Y = generate_dataset()
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
    classifier = clean_train_classifier(X_train, Y_train)

    csv_path = generate_filename(
        method=method,
        threshold=config["uncertainty_threshold"],
        base_dir="results",
    )

    evaluate_and_log_model(
        classifier,
        X_test,
        Y_test,
        csv_path,
        method,
        "pre",
        seed,
        threshold,
        gap_ratio=config["gap_ratio"],
    )

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_results_with_decision_boundary(
            classifier, X, Y, ax=axes[0], title="Pre-Gap"
        )
    except Exception as e:
        print(f"Error during visualization: {e}")

    # Assign gap class and augment
    Y_train_with_gap, _ = assign_gap_class(
        classifier, X_train, Y_train, threshold=config["uncertainty_threshold"]
    )

    labels_present = np.unique(Y_train_with_gap)
    print(f"Labels present after gap assignment: {labels_present}")

    unique_labels, counts = np.unique(Y_train_with_gap, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples")

    if method == "smote":
        X_aug, Y_aug = augment_smote_gap_class(
            X_train,
            Y_train_with_gap,
            target_class=config["gap_class_label"],
            gap_ratio=config["gap_ratio"],
        )
    elif method == "oversampling":
        X_aug, Y_aug = augment_oversampling_gap_class(
            X_train,
            Y_train_with_gap,
            target_class=config["gap_class_label"],
            gap_ratio=config["gap_ratio"],
        )
    elif method == "svm_smote":
        X_aug, Y_aug = augment_svm_smote_gap_class(
            X_train,
            Y_train_with_gap,
            target_class=config["gap_class_label"],
            gap_ratio=config["gap_ratio"],
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    classifier_aug = clean_train_classifier(X_aug, Y_aug)

    # Now plot the augmented version
    evaluate_and_log_model(
        classifier_aug,
        X_test,
        Y_test,
        csv_path,
        method,
        "post",
        seed,
        threshold,
        gap_ratio=config["gap_ratio"],
    )

    try:
        plot_results_with_decision_boundary(
            classifier_aug, X_aug, Y_aug, ax=axes[1], title="Post-Gap"
        )  # Y_with_gap
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error during visualization: {e}")


if __name__ == "__main__":
    main()
