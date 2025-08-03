from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from config import CONFIG
from dataset import generate_dataset, split_dataset
from enums import AugmentationMethod, Dataset
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

# === Configuration and Constants ===
UNCERTAINTY_THRESHOLD: float = CONFIG["uncertainty_threshold"]
RANDOM_STATE: int = CONFIG["random_state"]
GAP_RATIO: float = CONFIG["gap_ratio"]
GAP_CLASS_LABEL: int = CONFIG["gap_class_label"]
METHOD: AugmentationMethod = AugmentationMethod.SMOTE
CSV_PATH: str = generate_filename(METHOD, UNCERTAINTY_THRESHOLD, base_dir="results")
NUMBER_OF_CLASSES: int = 2

augmentation_dispatch = {
    AugmentationMethod.SMOTE: augment_smote_gap_class,
    AugmentationMethod.OVERSAMPLING: augment_oversampling_gap_class,
    AugmentationMethod.SVM_SMOTE: augment_svm_smote_gap_class,
}


def main() -> None:
    """
    Runs the complete experimental pipeline:
    - Prepares the data.
    - Trains and evaluates a baseline classifier.
    - Identifies low-confidence samples and assigns them to a gap class.
    - Augments the gap class using a chosen augmentation method.
    - Retrains the classifier and evaluates post-augmentation performance.
    - Visualizes decision boundaries before and after augmentation.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    print(
        f"Run (seed={RANDOM_STATE}): threshold={UNCERTAINTY_THRESHOLD} & gap_ratio={round(GAP_RATIO, 2)}"
    )

    X_train, X_test, y_train, y_test = prepare_data()
    print("--- SPLIT INFO ---")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train counts:", np.bincount(y_train))
    print("y_test counts:", np.bincount(y_test))
    print("Checksum (X_train[:10]):", hash(X_train[:10].tobytes()))
    print("Checksum (y_train[:10]):", hash(y_train[:10].tobytes()))
    print("------------------")

    classifier = clean_train_classifier(X_train, y_train, CONFIG["random_state"])

    pre_classifier = classifier
    evaluate_and_visualize_baseline(
        classifier, X_test, y_test, X_train, y_train, axes[0]
    )

    y_train_with_gap = assign_and_log_gap_class(classifier, X_train, y_train)
    X_aug, y_aug, _ = augment_data(X_train, y_train_with_gap)
    # y_aug = np.where(y_aug >= CONFIG["first_gap_class_label"], GAP_CLASS_LABEL, y_aug)

    classifier_aug = clean_train_classifier(X_aug, y_aug, CONFIG["random_state"])
    evaluate_and_visualize_augmented(
        classifier_aug, X_test, y_test, X_aug, y_aug, axes[1], pre_classifier
    )
    evaluate_on_test_data(classifier_aug, X_test, y_test)

    plt.tight_layout()
    plt.show()


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates and splits a synthetic dataset.

    Returns:
        Tuple containing training and test splits:
        (X_train, X_test, y_train, y_test)
    """

    X, y = generate_dataset(Dataset.MULTI_BLOBS)

    print("=== DATASET INFO ===")
    print("X shape:", X.shape)
    print("Y shape:", y.shape)
    print("Label counts:", np.bincount(y))
    print("X mean (per dim):", np.mean(X, axis=0))
    print("X std (per dim):", np.std(X, axis=0))
    print("First 5 samples (X):", X[:5])
    print("First 5 labels (Y):", y[:5])
    print("Checksum (X):", hash(X[:10].tobytes()))
    print("Checksum (Y):", hash(y[:10].tobytes()))
    print("=====================")

    NUMBER_OF_CLASSES = len(np.unique(y))
    print(f"Number of classes in dataset: {NUMBER_OF_CLASSES}")
    return split_dataset(X, y)


def evaluate_and_visualize_baseline(
    classifier, X_test, y_test, X_train, y_train, ax
) -> None:
    """
    Evaluates the baseline classifier before augmentation and plots its decision boundary.

    Args:
        classifier: Trained classifier to evaluate.
        X_test: Test features.
        y_test: Test labels.
        X_train: Training features.
        y_train: Training labels.
        ax: Matplotlib axis to draw the decision boundary on.
    """

    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[PRE-GAP] Accuracy: {acc:.4f}")

    evaluate_and_log_model(
        classifier=classifier,
        X_test=X_test,
        Y_test=y_test,
        file_path=CSV_PATH,
        method=METHOD.value,
        stage="pre",
        seed=RANDOM_STATE,
        threshold=UNCERTAINTY_THRESHOLD,
        gap_ratio=GAP_RATIO,
        pre_classifier=None,
    )

    try:
        plot_results_with_decision_boundary(
            classifier, X_train, y_train, ax=ax, title="Pre-Gap"
        )
    except Exception as e:
        print(f"Error during visualization: {e}")


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


def augment_data(X_train, y_train_with_gap) -> Tuple[np.ndarray, np.ndarray]:
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

    if NUMBER_OF_CLASSES <= 2:
        return augment_fn(
            X_train,
            y_train_with_gap,
            target_class=GAP_CLASS_LABEL,
            gap_ratio=GAP_RATIO,
        )
    else:
        first_gap_class_label = CONFIG["first_gap_class_label"]

        X_aug_total = [X_train]
        y_aug_total = [y_train_with_gap]

        for i in range(NUMBER_OF_CLASSES):
            gap_label = first_gap_class_label + i
            if gap_label not in y_train_with_gap:
                continue  # Skip if this gap label wasn't actually used

            X_gap, y_gap = augment_fn(
                X_train,
                y_train_with_gap,
                target_class=gap_label,
                gap_ratio=GAP_RATIO,
            )
            X_aug_total.append(X_gap)
            y_aug_total.append(y_gap)

        return np.vstack(X_aug_total), np.hstack(y_aug_total)


def evaluate_and_visualize_augmented(
    classifier, X_test, y_test, X_aug, y_aug, ax, pre_classifier=None
) -> None:
    """
    Evaluates the classifier after augmentation and visualizes the new decision boundary.

    Args:
        classifier: Retrained classifier after augmentation.
        X_test: Test features.
        y_test: Test labels.
        X_aug: Augmented training features.
        y_aug: Augmented training labels.
        ax: Matplotlib axis to draw the decision boundary on.
    """

    evaluate_and_log_model(
        classifier=classifier,
        X_test=X_test,
        Y_test=y_test,
        file_path=CSV_PATH,
        method=METHOD.value,
        stage="post",
        seed=RANDOM_STATE,
        threshold=UNCERTAINTY_THRESHOLD,
        gap_ratio=GAP_RATIO,
        pre_classifier=pre_classifier,
    )

    try:
        plot_results_with_decision_boundary(
            classifier, X_aug, y_aug, ax=ax, title="Post-Gap"
        )
    except Exception as e:
        print(f"Error during visualization: {e}")


def evaluate_on_test_data(classifier, X_test, y_test):
    from model import sanitize_predictions

    y_pred = classifier.predict(X_test)
    y_pred = sanitize_predictions(y_pred, valid_labels=np.unique(y_test))

    acc = accuracy_score(y_test, y_pred)
    print(f"[POST GAP (ON TEST DATA)] Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
