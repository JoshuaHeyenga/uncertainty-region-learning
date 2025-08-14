from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, BorderlineSMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

from config import CONFIG
from logger import log_metrics_to_csv


def clean_train_classifier(X_train, Y_train, seed):
    """
    Trains a multi-layer perceptron (MLP) classifier on the given training data.

    Args:
        X_train (ndarray): The feature matrix for training, shape (n_samples, n_features).
        Y_train (ndarray): The corresponding label vector, shape (n_samples,).

    Returns:
        MLPClassifier: The trained classifier.
    """

    classifier = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation="relu",
        solver="adam",
        max_iter=10000,
        random_state=seed,
        early_stopping=True,
    )
    classifier.fit(X_train, Y_train)
    return classifier


def evaluate_and_log_model(
    classifier,
    X_test,
    Y_test,
    file_path,
    method,
    stage,
    seed,
    threshold,
    gap_ratio,
    pre_classifier,
):
    y_pred = classifier.predict(X_test)

    valid_classes = np.unique(Y_test)
    y_pred = sanitize_predictions(y_pred, valid_classes)

    acc = accuracy_score(Y_test, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        Y_test, y_pred, labels=np.unique(Y_test), zero_division=0
    )

    if stage == "post":
        gap_clarity_gain = compute_gap_clarity_gain(
            pre_classifier=pre_classifier,
            post_classifier=classifier,
            X_test=X_test,
            y_test=Y_test,
            threshold=threshold,
            class_count=len(np.unique(Y_test)),
        )
    else:
        gap_clarity_gain = None

    for class_label, p, r, f, s in zip(
        np.unique(Y_test), precision, recall, f1, support
    ):
        if gap_clarity_gain == 0.0:
            continue
        else:
            log_metrics_to_csv(
                file_path=file_path,
                method=method,
                stage=stage,
                seed=seed,
                class_label=class_label,
                threshold=threshold,
                gap_ratio=gap_ratio,
                precision=p,
                recall=r,
                f1=f,
                support=s,
                accuracy=acc,
                gcg=gap_clarity_gain,
            )


def sanitize_predictions(y_pred, valid_labels):
    """
    Replace any predictions outside of valid labels with the most frequent label.
    """

    majority_class = Counter(valid_labels).most_common(1)[0][0]
    return np.array([y if y in valid_labels else majority_class for y in y_pred])


def compute_gap_clarity_gain(  # needs to be updated
    pre_classifier,
    post_classifier,
    X_test,
    y_test,
    threshold=CONFIG["uncertainty_threshold"],
    class_count: int = 2,
):
    pre_proba = pre_classifier.predict_proba(X_test)
    post_proba = post_classifier.predict_proba(X_test)

    base_classes = list(range(class_count))

    conf_before = np.max(pre_proba[:, base_classes], axis=1)
    conf_after = np.max(post_proba[:, base_classes], axis=1)

    boundary_mask = conf_before < (1 - threshold)

    if np.sum(boundary_mask) == 0:
        return None

    avg_conf_before = np.median(conf_before[boundary_mask])
    avg_conf_after = np.median(conf_after[boundary_mask])

    gcg_norm = (avg_conf_after - avg_conf_before) / (1 - avg_conf_before)
    return gcg_norm


def assign_gap_class(
    classifier, X, Y, threshold=CONFIG["uncertainty_threshold"], class_count: int = 2
):
    if threshold is None:
        threshold = threshold

    Y_extended = np.copy(Y)
    print("Classes in Y pre-augmentation:", np.unique(Y_extended))

    if class_count <= 2:
        print("Assigning gap class for binary classification.")
        proba = classifier.predict_proba(X)
        confidence = np.max(proba, axis=1)

        uncertain_mask = confidence < (1 - threshold)
        Y_extended[uncertain_mask] = CONFIG["gap_class_label"]
        return Y_extended, uncertain_mask
    else:
        # print("Assigning gap class for multi-class classification.")

        gap_class_label = CONFIG["first_gap_class_label"]
        partial_gap_masks = {}
        proba_all = classifier.predict_proba(X)
        loop_helper = 0

        unique_classes = np.unique(Y)
        # print("Unique classes in Y:", unique_classes)
        for cls in unique_classes:
            # print(f"\n--- Class {cls} ---")
            # print(f"Loop helper value: {loop_helper}")
            proba_cls = proba_all[:, loop_helper]
            proba_not_cls = np.sum(np.delete(proba_all, loop_helper, axis=1), axis=1)

            # Consider samples where the true label is cls
            belongs_to_class = Y == cls
            # print(f"Total samples belonging to class {cls}: {np.sum(belongs_to_class)}")

            # Low confidence that it's either cls or not-cls
            confidence = np.maximum(proba_cls, proba_not_cls)
            # below_threshold = confidence < (1 - threshold)  # optional
            uncertain_mask = (confidence < (1 - threshold)) & belongs_to_class
            # print(f"Samples below threshold for class {cls}: {np.sum(below_threshold)}")
            # print(
            #    f"Uncertain samples (final selection) for class {cls}: {np.sum(uncertain_mask)}"
            # )

            partial_gap_masks[cls] = uncertain_mask
            Y_extended[uncertain_mask] = gap_class_label
            # print(
            #    f"Assigned gap label {gap_class_label} for class {cls} — {np.sum(uncertain_mask)} samples."
            # )
            # print("Classes in Y post assignmend (in loop):", np.unique(Y_extended))
            gap_class_label += 1
            loop_helper += 1

        return Y_extended, partial_gap_masks


def augment_oversampling_gap_class(
    X, Y, target_class=CONFIG["gap_class_label"], gap_ratio=CONFIG["gap_ratio"]
):
    _, target_count, _, _, _ = get_gap_class_target_count(
        Y, target_class, ratio=gap_ratio
    )

    # if not needs_aug:
    # print("No oversampling needed.")
    # return X, Y, False

    n_existing = np.sum(Y == target_class)
    # n_to_generate = target_count - n_existing

    X_target = X[Y == target_class]
    Y_target = Y[Y == target_class]
    n_existing = len(Y_target)

    if n_existing < 2:
        print("Not enough uncertainty samples. Skipping augmentation.")
        return X, Y, False

    X_oversampled, Y_oversampled = resample(
        X_target,
        Y_target,
        replace=True,
        n_samples=target_count,  # originally n_to_generate
        random_state=CONFIG["random_state"],
    )

    # Concatenate original and new samples
    X_augmented = np.vstack((X, X_oversampled))
    Y_augmented = np.hstack((Y, Y_oversampled))

    return X_augmented, Y_augmented, True


def augment_smote_gap_class(
    X, Y, target_class=CONFIG["gap_class_label"], gap_ratio=CONFIG["gap_ratio"]
):
    _, target_count, _, _, _ = get_gap_class_target_count(
        Y, target_class, ratio=gap_ratio
    )

    n_existing = np.sum(Y == target_class)
    total_count = n_existing + target_count
    if n_existing < 2:
        print("Not enough uncertainty samples. Skipping augmentation.")
        return X, Y, False

    try:
        smoter = SMOTE(
            sampling_strategy={target_class: total_count},
            random_state=CONFIG["random_state"],
        )
        X_aug, Y_aug = smoter.fit_resample(X, Y)

    except ValueError as e:
        print(f"SMOTE failed: {e}")
        return X, Y, False

    return X_aug, Y_aug, True


def augment_svm_smote_gap_class(
    X, Y, target_class=CONFIG["gap_class_label"], gap_ratio=CONFIG["gap_ratio"]
):
    _, target_count, _, _, _ = get_gap_class_target_count(
        Y, target_class, ratio=gap_ratio
    )

    n_existing = np.sum(Y == target_class)
    total_count = n_existing + target_count
    if n_existing < 2:
        print("Not enough uncertainty samples. Skipping augmentation.")
        return X, Y, False

    try:
        smoter = SVMSMOTE(
            sampling_strategy={target_class: total_count},
            random_state=CONFIG["random_state"],
        )
        X_aug, Y_aug = smoter.fit_resample(X, Y)
    except ValueError as e:
        print(f"SVMSMOTE failed: {e}")
        return X, Y, False

    return X_aug, Y_aug, True


def augment_borderline_smote_gap_class(
    X, Y, target_class=CONFIG["gap_class_label"], gap_ratio=CONFIG["gap_ratio"]
):
    _, target_count, _, _, _ = get_gap_class_target_count(
        Y, target_class, ratio=gap_ratio
    )

    n_existing = np.sum(Y == target_class)
    total_count = n_existing + target_count
    if n_existing < 2:
        print("Not enough uncertainty samples. Skipping augmentation.")
        return X, Y, False

    try:
        smoter = BorderlineSMOTE(
            sampling_strategy={target_class: total_count},
            random_state=CONFIG["random_state"],
        )
        X_aug, Y_aug = smoter.fit_resample(X, Y)
    except ValueError as e:
        print(f"BorderlineSMOTE failed: {e}")
        return X, Y, False

    return X_aug, Y_aug, True


def augment_adasyn_gap_class(
    X, Y, target_class=CONFIG["gap_class_label"], gap_ratio=CONFIG["gap_ratio"]
):
    _, target_count, _, _, _ = get_gap_class_target_count(
        Y, target_class, ratio=gap_ratio
    )

    n_existing = np.sum(Y == target_class)
    total_count = n_existing + target_count
    if n_existing < 2:
        print("Not enough uncertainty samples. Skipping augmentation.")
        return X, Y, False

    try:
        adasyn = ADASYN(
            sampling_strategy={target_class: total_count},
            random_state=CONFIG["random_state"],
        )
        X_aug, Y_aug = adasyn.fit_resample(X, Y)
    except ValueError as e:
        print(f"ADASYN failed: {e}")
        return X, Y, False

    return X_aug, Y_aug, True


def get_gap_class_target_count(
    Y, target_class=CONFIG["gap_class_label"], ratio=CONFIG["gap_ratio"]
):
    class_counts = {
        label: np.sum(Y == label) for label in np.unique(Y) if label != target_class
    }

    avg_class_size = np.mean(list(class_counts.values()))
    non_gap_labels = list(class_counts.keys())

    target_count = int(ratio * avg_class_size)
    current_gap_count = np.sum(Y == target_class)

    needs_augmentation = current_gap_count < target_count
    return (
        needs_augmentation,
        target_count,
        avg_class_size,
        current_gap_count,
        non_gap_labels,
    )
