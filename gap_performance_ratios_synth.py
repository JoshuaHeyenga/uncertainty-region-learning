import os
from datetime import datetime
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score

from config import CONFIG
from dataset import generate_dataset, split_dataset
from enums import AugmentationMethod, Dataset, PerformanceMetric, PerformanceStage
from model import (
    assign_gap_class,
    augment_oversampling_gap_class,
    augment_smote_gap_class,
    augment_svm_smote_gap_class,
    clean_train_classifier,
    evaluate_and_log_model,
)
from visualization import (
    plot_gcg_across_ratios,
    plot_performance_across_ratios,
    plot_std_performance,
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
SYNTH_DIR: str = "results/final_results/"
TIMESTAMP: str = datetime.now().strftime("%m.%d_%H.%M")
FILE_NAME: str = f"ts_ratio_results__{TIMESTAMP}.csv"
FILE_PATH: str = os.path.join(SYNTH_DIR, FILE_NAME)

MANUAL_FILE_PATH: str = "results/final_results/ts_ratio_results__08.03_10.00.csv"

PRE_CLASSIFIER = None

augmentation_dispatch = {
    AugmentationMethod.SMOTE: augment_smote_gap_class,
}


def main() -> None:
    for seed in RANDOM_STATES:
        get_seed_performance(seed=seed)


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

            classifier = clean_train_classifier(X_train, y_train, seed)

            y_pred = classifier.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"[PRE-GAP] Accuracy: {acc:.4f}")

            global PRE_CLASSIFIER
            PRE_CLASSIFIER = classifier

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
                pre_classifier=None,
            )

            # Save original labels
            original_y_train = y_train.copy()

            # == Get Gap-Class Performance ==
            y_train_gap, _ = assign_gap_class(
                classifier=classifier,
                X=X_train,
                Y=y_train,
                threshold=threshold,
            )

            X_aug, y_aug, was_augmented = augment_data(X_train, y_train_gap, gap_ratio)

            if not was_augmented:
                print("Skipping retraining and logging — no augmentation needed.")
                continue

            y_aug[: len(y_train)] = original_y_train  # needs further testing

            classifier_aug = clean_train_classifier(X_aug, y_aug, seed)

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
                pre_classifier=PRE_CLASSIFIER,
            )


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates and splits a synthetic dataset.

    Returns:
        Tuple containing training and test splits:
        (X_train, X_test, y_train, y_test)
    """

    X, y = generate_dataset(mode=Dataset.BLOBS)
    return split_dataset(X, y)


def augment_data(
    X_train, y_train_with_gap, gap_ratio
) -> Tuple[np.ndarray, np.ndarray, bool]:
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

    X_aug, y_aug, was_augmented = augment_fn(
        X_train,
        y_train_with_gap,
        target_class=FIRST_GAP_CLASS_LABEL,
        gap_ratio=gap_ratio,
    )

    return X_aug, y_aug, was_augmented


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


if __name__ == "__main__":
    plot_std_performance(
        MANUAL_FILE_PATH,
        0,
        PerformanceMetric.ACCURACY.value,
        PerformanceStage.POST.value,
        True,
    )
    plot_std_performance(
        MANUAL_FILE_PATH,
        1,
        PerformanceMetric.ACCURACY.value,
        PerformanceStage.POST.value,
        True,
    )
    plot_std_performance(
        MANUAL_FILE_PATH,
        0,
        PerformanceMetric.F1SCORE.value,
        PerformanceStage.POST.value,
        True,
    )
    plot_std_performance(
        MANUAL_FILE_PATH,
        1,
        PerformanceMetric.F1SCORE.value,
        PerformanceStage.POST.value,
        True,
    )
