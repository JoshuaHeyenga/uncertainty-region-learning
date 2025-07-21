import os
from datetime import datetime
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score

from config import CONFIG
from dataset import generate_dataset, split_dataset
from enums import AugmentationMethod, Dataset
from model import (
    assign_gap_class,
    augment_adasyn_gap_class,
    augment_borderline_smote_gap_class,
    augment_oversampling_gap_class,
    augment_smote_gap_class,
    augment_svm_smote_gap_class,
    clean_train_classifier,
    evaluate_and_log_model,
)
from visualization import plot_gcg, plot_performance_across_thresholds

# === CONFIG ===
# General Data
UNCERTAINTY_THRESHOLD: float = CONFIG["uncertainty_threshold"]
RANDOM_STATES: int = CONFIG["random_states"]
FIRST_GAP_CLASS_LABEL: int = CONFIG["first_gap_class_label"]
NUMBER_OF_CLASSES: int

# Testing Range
THRESHOLDS: float = CONFIG["threshold_performance_thresholds"]
GAP_RATIOS: float = CONFIG["threshold_performance_gap_ratios"]

# Logging
SYNTH_DIR: str = "results/synth_dataset/"
TIMESTAMP: str = datetime.now().strftime("%m.%d_%H.%M")
FILE_NAME: str = f"two_synth_results_all-methods_{TIMESTAMP}.csv"
FILE_PATH: str = os.path.join(SYNTH_DIR, FILE_NAME)
MANUAL_FILE_PATH: str = (
    "results/synth_dataset/two_synth_results_all-methods_07.15_17.25.csv"
)

PRE_CLASSIFIER = None

augmentation_dispatch = {
    AugmentationMethod.OVERSAMPLING: augment_oversampling_gap_class,
    AugmentationMethod.SMOTE: augment_smote_gap_class,
    AugmentationMethod.SVM_SMOTE: augment_svm_smote_gap_class,
    AugmentationMethod.BORDERLINE_SMOTE: augment_borderline_smote_gap_class,
    AugmentationMethod.ADASYN: augment_adasyn_gap_class,
}


def main() -> None:
    for method in augmentation_dispatch:
        for seed in RANDOM_STATES:
            get_seed_performance_for_method(seed=seed, augment_method=method)


def get_seed_performance_for_method(seed: int, augment_method: AugmentationMethod):
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
                method=augment_method.value,
                stage="pre",
                seed=seed,
                threshold=threshold,
                gap_ratio=gap_ratio,
                pre_classifier=None,
            )

            # == Get Gap-Class Performance ==
            y_train_gap, _ = assign_gap_class(
                classifier=classifier,
                X=X_train,
                Y=y_train,
                threshold=threshold,
                class_count=NUMBER_OF_CLASSES,
            )

            X_aug, y_aug = augment_data(X_train, y_train_gap, gap_ratio, augment_method)
            y_aug = np.where(
                y_aug >= CONFIG["first_gap_class_label"], FIRST_GAP_CLASS_LABEL, y_aug
            )
            classifier_aug = clean_train_classifier(X_aug, y_aug, seed)

            evaluate_and_log_model(
                classifier=classifier_aug,
                X_test=X_test,
                Y_test=y_test,
                file_path=FILE_PATH,
                method=augment_method.value,
                stage="post",
                seed=seed,
                threshold=threshold,
                gap_ratio=gap_ratio,
                pre_classifier=classifier,
            )


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates and splits a synthetic dataset.

    Returns:
        Tuple containing training and test splits:
        (X_train, X_test, y_train, y_test)
    """

    X, y = generate_dataset(Dataset.BLOBS)

    global NUMBER_OF_CLASSES
    NUMBER_OF_CLASSES = len(np.unique(y))
    print(f"Number of classes in dataset: {NUMBER_OF_CLASSES}")

    return split_dataset(X, y)


def assign_and_log_gap_class(classifier, X_train, y_train) -> np.ndarray:
    """
    Identifies uncertain training samples and assigns them to the gap class.

    Args:
        classifier: Trained classifier used for confidence evaluation.
        X_train: Training features.
        y_train: Original training labels.

    Returns:
        Modified training labels with low-confidence points relabeled as gap class.
    """

    y_train_with_gap, partial_gap_masks = assign_gap_class(
        classifier,
        X_train,
        y_train,
        threshold=UNCERTAINTY_THRESHOLD,
        class_count=NUMBER_OF_CLASSES,
    )

    gap_label = CONFIG["first_gap_class_label"]
    gap_indices = y_train_with_gap == gap_label
    original_labels_of_gap_samples = y_train[gap_indices]

    print(
        f"Original labels of gap samples: {np.unique(original_labels_of_gap_samples)}"
    )

    return y_train_with_gap


def augment_data(
    X_train, y_train_with_gap, gap_ratio, augment_method: AugmentationMethod
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the selected augmentation method to the gap class to balance it.

    Args:
        X_train: Training features.
        y_train_with_gap: Training labels including gap class.

    Returns:
        Tuple of (X_aug, y_aug): Augmented feature and label arrays.
    """

    augment_fn = augmentation_dispatch.get(augment_method)
    if not augment_fn:
        raise ValueError(f"Unsupported augmentation method: {augment_method.value}")

    return augment_fn(
        X_train,
        y_train_with_gap,
        target_class=FIRST_GAP_CLASS_LABEL,
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


if __name__ == "__main__":
    plot_performance_across_thresholds(
        MANUAL_FILE_PATH, AugmentationMethod.ADASYN.value, 0.3
    )
    plot_performance_across_thresholds(
        MANUAL_FILE_PATH, AugmentationMethod.ADASYN.value, 0.9
    )
    plot_performance_across_thresholds(
        MANUAL_FILE_PATH, AugmentationMethod.ADASYN.value, 1.5
    )
