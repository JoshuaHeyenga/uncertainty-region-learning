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
from visualization import plot_performance_accross_ratios, plot_std_performance

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
SYNTH_DIR: str = "results/synth_dataset/"
TIMESTAMP: str = datetime.now().strftime("%m.%d_%H.%M")
FILE_NAME: str = f"synth_results_{METHOD.value}_{TIMESTAMP}.csv"
FILE_PATH: str = os.path.join(SYNTH_DIR, FILE_NAME)

PRE_CLASSIFIER = None

MANUAL_FILE_PATH: str = "results/synth_dataset/synth_results_smote_07.21_17.46.csv"

augmentation_dispatch = {
    AugmentationMethod.SMOTE: augment_smote_gap_class,
    AugmentationMethod.OVERSAMPLING: augment_oversampling_gap_class,
    AugmentationMethod.SVM_SMOTE: augment_svm_smote_gap_class,
}


def main() -> None:
    for seed in RANDOM_STATES:
        get_seed_performance(seed=seed)

    plot_performance_accross_ratios(
        file_path=FILE_PATH,
        obs_class=1,
        n_cols=2,
        n_rows=2,
    )


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

            X_aug, y_aug, was_augmented = augment_data(X_train, y_train_gap, gap_ratio)

            if not was_augmented:
                print("Skipping retraining and logging — no augmentation needed.")
                continue

            y_aug = np.where(
                y_aug >= CONFIG["first_gap_class_label"], FIRST_GAP_CLASS_LABEL, y_aug
            )
            y_aug[: len(y_train)] = original_y_train

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
    X_train, y_train_with_gap, gap_ratio
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
        X_aug_total = [X_train]
        y_aug_total = [y_train_with_gap]
        augmented = False

        for i in range(NUMBER_OF_CLASSES):
            gap_label = FIRST_GAP_CLASS_LABEL + i
            if gap_label not in y_train_with_gap:
                continue

            X_gap, y_gap, was_aug = augment_fn(
                X_train,
                y_train_with_gap,
                target_class=gap_label,
                gap_ratio=gap_ratio,
            )
            if was_aug:
                augmented = True
                X_aug_total.append(X_gap)
                y_aug_total.append(y_gap)

        return np.vstack(X_aug_total), np.hstack(y_aug_total), augmented


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
        2,
        PerformanceMetric.ACCURACY.value,
        PerformanceStage.POST.value,
        True,
    )
    plot_std_performance(
        MANUAL_FILE_PATH,
        3,
        PerformanceMetric.ACCURACY.value,
        PerformanceStage.POST.value,
        True,
    )
