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

# Testing Range
THRESHOLDS: float = CONFIG["synth_thresholds"]
GAP_RATIOS: float = CONFIG["gap_ratios"]

# Logging
SYNTH_DIR: str = "results/synth_dataset/"
TIMESTAMP: str = datetime.now().strftime("%m.%d_%H.%M")
FILE_NAME: str = f"synth_results_{METHOD.value}_{TIMESTAMP}.csv"
FILE_PATH: str = os.path.join(SYNTH_DIR, FILE_NAME)

MANUAL_FILE_PATH: str = "results/synth_dataset/synth_results_smote_07.10_16.20.csv"

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
            )


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates and splits a synthetic dataset.

    Returns:
        Tuple containing training and test splits:
        (X_train, X_test, y_train, y_test)
    """

    X, y = generate_dataset(mode=Dataset.MULTI_BLOBS)
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


if __name__ == "__main__":
    plot_performance_accross_ratios(
        file_path=MANUAL_FILE_PATH,
        obs_class=1,
        n_cols=2,
        n_rows=2,
    )

    # F1 Score
    plot_std_performance(
        file_path=MANUAL_FILE_PATH,
        obs_class=0,
        metric=PerformanceMetric.F1SCORE.value,
        stage=PerformanceStage.POST.value,
        comp=True,
    )
    plot_std_performance(
        file_path=MANUAL_FILE_PATH,
        obs_class=1,
        metric=PerformanceMetric.F1SCORE.value,
        stage=PerformanceStage.POST.value,
        comp=True,
    )

    # Recall
    plot_std_performance(
        file_path=MANUAL_FILE_PATH,
        obs_class=0,
        metric=PerformanceMetric.RECALL.value,
        stage=PerformanceStage.POST.value,
        comp=True,
    )
    plot_std_performance(
        file_path=MANUAL_FILE_PATH,
        obs_class=1,
        metric=PerformanceMetric.RECALL.value,
        stage=PerformanceStage.POST.value,
        comp=True,
    )

    # Accuracy
    plot_std_performance(
        file_path=MANUAL_FILE_PATH,
        obs_class=0,
        metric=PerformanceMetric.ACCURACY.value,
        stage=PerformanceStage.POST.value,
        comp=True,
    )
    plot_std_performance(
        file_path=MANUAL_FILE_PATH,
        obs_class=1,
        metric=PerformanceMetric.ACCURACY.value,
        stage=PerformanceStage.POST.value,
        comp=True,
    )

    # Precision
    plot_std_performance(
        file_path=MANUAL_FILE_PATH,
        obs_class=0,
        metric=PerformanceMetric.PRECISION.value,
        stage=PerformanceStage.POST.value,
        comp=True,
    )
    plot_std_performance(
        file_path=MANUAL_FILE_PATH,
        obs_class=1,
        metric=PerformanceMetric.PRECISION.value,
        stage=PerformanceStage.POST.value,
        comp=True,
    )
