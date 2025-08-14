import os
from datetime import datetime
from typing import Tuple

import numpy as np

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
from visualization import plot_performance_across_ratios, plot_std_performance

# === CONFIG ===
# General Data
UNCERTAINTY_THRESHOLD: float = CONFIG["uncertainty_threshold"]
RANDOM_STATES: int = CONFIG["random_states"]
FIRST_GAP_CLASS_LABEL: int = CONFIG["first_gap_class_label"]
METHOD: AugmentationMethod = AugmentationMethod.SMOTE
NUMBER_OF_CLASSES: int

# Testing Range
THRESHOLDS: float = CONFIG["synth_thresholds"]
GAP_RATIOS: float = CONFIG["gap_ratios"]

# Logging
SYNTH_DIR: str = "results/final_results/"
TIMESTAMP: str = datetime.now().strftime("%m.%d_%H.%M")
FILE_NAME: str = f"ms_ratio_results__{TIMESTAMP}.csv"
FILE_PATH: str = os.path.join(SYNTH_DIR, FILE_NAME)

PRE_CLASSIFIER = None

MANUAL_FILE_PATH: str = "results/final_results/ms_ratio_results__08.06_12.19.csv"

augmentation_dispatch = {
    AugmentationMethod.SMOTE: augment_smote_gap_class,
}


def main() -> None:
    for seed in RANDOM_STATES:
        get_seed_performance(seed=seed)

    print("=== Performance Evaluation Completed ===")


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

            original_y_train = y_train.copy()

            # == Get Gap-Class Performance ==
            y_train_gap, _ = assign_gap_class(
                classifier=classifier,
                X=X_train,
                Y=y_train,
                threshold=threshold,
                class_count=NUMBER_OF_CLASSES,
            )

            gap_labels = [
                label
                for label in np.unique(y_train_gap)
                if label >= FIRST_GAP_CLASS_LABEL
            ]
            num_gap_assigned = np.sum(np.isin(y_train_gap, gap_labels))

            if num_gap_assigned == 0:
                print("Empty gap class, skipping augmentation.")
                continue

            X_aug, y_aug, was_augmented = augment_data(
                X_train, y_train_gap, original_y_train, gap_ratio
            )

            if not was_augmented:
                print("Skipping retraining and logging — no augmentation needed.")
                continue

            y_aug = np.where(
                y_aug >= CONFIG["first_gap_class_label"], FIRST_GAP_CLASS_LABEL, y_aug
            )

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

    X, y = generate_dataset(mode=Dataset.MULTI_BLOBS)
    global NUMBER_OF_CLASSES
    NUMBER_OF_CLASSES = len(np.unique(y))
    return split_dataset(X, y)


def augment_data(
    X_train, y_train_with_gap, original_y_train, gap_ratio
) -> Tuple[np.ndarray, np.ndarray, bool]:
    augment_fn = augmentation_dispatch.get(METHOD)
    if not augment_fn:
        raise ValueError(f"Unsupported augmentation method: {METHOD}")

    if NUMBER_OF_CLASSES <= 2:
        X_aug, y_aug, was_augmented = augment_fn(
            X_train,
            y_train_with_gap,
            target_class=FIRST_GAP_CLASS_LABEL,
            gap_ratio=gap_ratio,
        )
        return X_aug, y_aug, was_augmented
    else:
        X_aug = X_train.copy()
    y_aug = y_train_with_gap.copy()
    was_augmented = False

    gap_labels = sorted(
        [
            label
            for label in np.unique(y_train_with_gap)
            if label >= FIRST_GAP_CLASS_LABEL
        ]
    )

    for gap_label in gap_labels:
        print(f"[augment_data] Attempting to augment class: {gap_label}")
        original_count = np.sum(y_train_with_gap == gap_label)
        print(
            f"[augment_data] Number of original gap samples for {gap_label}: {original_count}"
        )

        # Augmentiere nur diese Partial Gap Klasse
        X_temp, y_temp, augmented = augment_fn(
            X_train,
            y_train_with_gap,
            target_class=gap_label,
            gap_ratio=gap_ratio,
        )

        if augmented:
            # Nur die neuen Samples herausziehen
            new_samples = X_temp[len(X_train) :]
            new_labels = np.full(len(new_samples), FIRST_GAP_CLASS_LABEL)

            X_aug = np.vstack((X_aug, new_samples))
            y_aug = np.hstack((y_aug, new_labels))
            was_augmented = True

    # Original-Gap-Samples wiederherstellen
    y_aug[: len(original_y_train)] = original_y_train

    return X_aug, y_aug, was_augmented


if __name__ == "__main__":
    plot_performance_across_ratios(MANUAL_FILE_PATH, 0)
    plot_performance_across_ratios(MANUAL_FILE_PATH, 1)
    plot_performance_across_ratios(MANUAL_FILE_PATH, 2)
    plot_performance_across_ratios(MANUAL_FILE_PATH, 3)
