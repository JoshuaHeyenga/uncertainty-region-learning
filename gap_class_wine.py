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
from visualization import (
    plot_gcg,
    plot_gcg_across_ratios,
    plot_performance_across_thresholds,
)

# === CONFIG ===
# General Data
UNCERTAINTY_THRESHOLD: float = CONFIG["uncertainty_threshold"]
RANDOM_STATES: int = CONFIG["random_states"]
FIRST_GAP_CLASS_LABEL: int = CONFIG["first_gap_class_label"]
NUMBER_OF_CLASSES: int

# Testing Range
THRESHOLDS: float = CONFIG["wine_thresholds"]
GAP_RATIOS: float = CONFIG["threshold_performance_gap_ratios"]

# Logging
SYNTH_DIR: str = "results/final_results/"
TIMESTAMP: str = datetime.now().strftime("%m.%d_%H.%M")
FILE_NAME: str = f"wine_results_all-methods__{TIMESTAMP}.csv"
FILE_PATH: str = os.path.join(SYNTH_DIR, FILE_NAME)
MANUAL_FILE_PATH: str = "results/final_results/ms_results_all-methods__08.03_17.49.csv"

PRE_CLASSIFIER = None

augmentation_dispatch = {
    AugmentationMethod.OVERSAMPLING: augment_oversampling_gap_class,
    AugmentationMethod.SMOTE: augment_smote_gap_class,
    AugmentationMethod.SVM_SMOTE: augment_svm_smote_gap_class,
    AugmentationMethod.BORDERLINE_SMOTE: augment_borderline_smote_gap_class,
    AugmentationMethod.ADASYN: augment_adasyn_gap_class,
}


def main() -> None:
    """X_train, X_test, y_train, y_test = prepare_data()
    classifier = clean_train_classifier(X_train, y_train, 42)
    count_uncertainty_samples_over_thresholds(
        classifier=classifier, X=X_train, Y=y_train
    )"""

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

            global PRE_CLASSIFIER
            PRE_CLASSIFIER = classifier

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
                X_train, y_train_gap, original_y_train, gap_ratio, augment_method
            )

            if not was_augmented:
                print("Skipping retraining and logging — no augmentation needed.")
                continue

            y_aug = np.where(
                y_aug >= CONFIG["first_gap_class_label"], FIRST_GAP_CLASS_LABEL, y_aug
            )

            # y_aug[: len(y_train)] = original_y_train

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
                pre_classifier=PRE_CLASSIFIER,
            )


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates and splits a synthetic dataset.

    Returns:
        Tuple containing training and test splits:
        (X_train, X_test, y_train, y_test)
    """

    X, y = generate_dataset(mode=Dataset.WINE)
    global NUMBER_OF_CLASSES
    NUMBER_OF_CLASSES = len(np.unique(y))
    return split_dataset(X, y)


def augment_data(
    X_train,
    y_train_with_gap,
    original_y_train,
    gap_ratio,
    augment_method: AugmentationMethod,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    augment_fn = augmentation_dispatch.get(augment_method)
    if not augment_fn:
        raise ValueError(f"Unsupported augmentation method: {augment_method.value}")

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

    print(f"[augment_data] Augmentation method: {augment_method}")
    print(f"[augment_data] Gap labels to augment: {gap_labels}")

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


def count_uncertainty_samples_over_thresholds(classifier, X, Y, gap_label=99):
    threshold_range = np.arange(0.01, 1.01, 0.01)
    proba = classifier.predict_proba(X)
    confidence = np.max(proba, axis=1)

    threshold_to_count = {}
    for threshold in threshold_range:
        uncertain_mask = confidence < (1 - threshold)
        count = np.sum(uncertain_mask)
        threshold_to_count[round(threshold, 2)] = count
        print(f"Threshold: {round(threshold, 2)}, Count: {count}")


if __name__ == "__main__":
    main()
