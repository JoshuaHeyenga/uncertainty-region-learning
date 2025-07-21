import numpy as np
import yaml
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, BorderlineSMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

from config import CONFIG
from logger import log_metrics_to_csv

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


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
        activation="relu",  # tanh, logistic, relu
        solver="adam",
        max_iter=2000,
        random_state=seed,
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
        print(f"Gap Clarity Gain: {gap_clarity_gain:.4f}")
    else:
        gap_clarity_gain = None

    for class_label, p, r, f, s in zip(
        np.unique(Y_test), precision, recall, f1, support
    ):
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
    from collections import Counter

    majority_class = Counter(valid_labels).most_common(1)[0][0]
    return np.array([y if y in valid_labels else majority_class for y in y_pred])


def compute_gap_clarity_gain(
    pre_classifier,
    post_classifier,
    X_test,
    y_test,
    threshold=CONFIG["uncertainty_threshold"],
    class_count: int = 2,
):
    pre_proba = pre_classifier.predict_proba(X_test)
    post_proba = post_classifier.predict_proba(X_test)

    base_classes = [0, 1]

    conf_before = np.max(pre_proba[:, base_classes], axis=1)
    conf_after = np.max(post_proba[:, base_classes], axis=1)

    # print(f"Conf before: {conf_before}")
    # print(f"Conf after: {conf_after}")

    boundary_mask = conf_before < (1 - threshold)

    if np.sum(boundary_mask) == 0:
        # print("No boundary samples found. Cannot compute clarity gain.")
        return 0.0

    avg_conf_before = np.median(conf_before[boundary_mask])
    avg_conf_after = np.median(conf_after[boundary_mask])

    gcg_norm = (avg_conf_after - avg_conf_before) / (1 - avg_conf_before)
    return gcg_norm


def assign_gap_class(
    classifier, X, Y, threshold=config["uncertainty_threshold"], class_count: int = 2
):
    """
    Assigns class label 2 (gap class) to data points with low classification confidence.

    This is based on the maximum predicted class probability being below a threshold,
    defined in the config file under 'uncertainty_threshold'.

    Args:
        classifier (MLPClassifier): Trained classifier used to compute prediction probabilities.
        X (ndarray): Feature matrix for all data points, shape (n_samples, n_features).
        Y (ndarray): Original label vector, shape (n_samples,).

    Returns:
        tuple:
            - ndarray: Updated label vector with uncertain samples relabeled to class 2.
            - ndarray: Boolean mask indicating which points were labeled as uncertain.
    """

    if threshold is None:
        threshold = threshold

    Y_extended = np.copy(Y)

    if class_count <= 2:
        proba = classifier.predict_proba(X)
        confidence = np.max(proba, axis=1)

        uncertain_mask = confidence < (1 - threshold)
        Y_extended[uncertain_mask] = config["gap_class_label"]

        return Y_extended, uncertain_mask
    else:
        gap_class_label = CONFIG["first_gap_class_label"]
        partial_gap_masks = {}
        proba_all = classifier.predict_proba(X)

        for cls in range(class_count):
            proba_cls = proba_all[:, cls]
            proba_not_cls = np.sum(np.delete(proba_all, cls, axis=1), axis=1)

            # Consider samples where the true label is cls
            belongs_to_class = Y == cls

            # Low confidence that it's either cls or not-cls
            confidence = np.maximum(proba_cls, proba_not_cls)
            uncertain_mask = (confidence < (1 - threshold)) & belongs_to_class

            partial_gap_masks[cls] = uncertain_mask
            Y_extended[uncertain_mask] = gap_class_label
            gap_class_label += 1

        return Y_extended, partial_gap_masks


def augment_oversampling_gap_class(
    X, Y, target_class=config["gap_class_label"], gap_ratio=config["gap_ratio"]
):
    """
    Augments the dataset by oversampling the gap class (label 2) using basic oversampling.

    The gap class is duplicated to reach double its original count. If the current number
    of gap-class samples already meets or exceeds the target, no augmentation is performed.

    Args:
        X (ndarray): The feature matrix including original samples, shape (n_samples, n_features).
        Y (ndarray): The label vector with gap class assignments, shape (n_samples,).
        target_class (int, optional): The class to oversample. Default is 2.

    Returns:
        tuple:
            - ndarray: Augmented feature matrix with new synthetic gap samples.
            - ndarray: Corresponding label vector including labels for new samples.
    """
    needs_aug, target_count, _, _, _ = get_gap_class_target_count(
        Y, target_class, ratio=gap_ratio
    )

    if not needs_aug:
        print("No oversampling needed.")
        return X, Y

    n_existing = np.sum(Y == target_class)
    n_to_generate = target_count - n_existing

    X_target = X[Y == target_class]
    Y_target = Y[Y == target_class]

    X_oversampled, Y_oversampled = resample(
        X_target,
        Y_target,
        replace=True,
        n_samples=n_to_generate,
        random_state=config["random_state"],
    )

    # Concatenate original and new samples
    X_augmented = np.vstack((X, X_oversampled))
    Y_augmented = np.hstack((Y, Y_oversampled))

    return X_augmented, Y_augmented


def augment_smote_gap_class(
    X, Y, target_class=config["gap_class_label"], gap_ratio=config["gap_ratio"]
):
    """
    Augments the dataset by synthetically oversampling the gap class (label 2) using basic SMOTE.

    The gap class is duplicated to reach double its original count. If the current number
    of gap-class samples already meets or exceeds the target, no augmentation is performed.

    Args:
        X (ndarray): The feature matrix including original samples, shape (n_samples, n_features).
        Y (ndarray): The label vector with gap class assignments, shape (n_samples,).
        target_class (int, optional): The class to oversample. Default is 2.

    Returns:
        tuple:
            - ndarray: Augmented feature matrix with new synthetic gap samples.
            - ndarray: Corresponding label vector including labels for new samples.
    """

    needs_aug, target_count, avg_top_two, current_gap_size, top_two_labels = (
        get_gap_class_target_count(Y, target_class, ratio=gap_ratio)
    )

    if not needs_aug:
        print("No SMOTE needed.")
        return X, Y, False
    else:
        print(f"Augmenting gap class {target_class} to target count: {target_count}")

    smoter = SMOTE(
        sampling_strategy={target_class: target_count},
        random_state=config["random_state"],
    )
    X_aug, Y_aug = smoter.fit_resample(X, Y)

    return X_aug, Y_aug, True


def augment_svm_smote_gap_class(
    X, Y, target_class=config["gap_class_label"], gap_ratio=config["gap_ratio"]
):
    """
    Augments the dataset by synthetically oversampling the gap class (label 2)
    using SVM-SMOTE from imbalanced-learn.

    SVMSMOTE performs SMOTE on the support vectors near the decision boundary
    of an SVM trained on the minority class.

    Args:
        X (ndarray): The feature matrix including original samples, shape (n_samples, n_features).
        Y (ndarray): The label vector with gap class assignments, shape (n_samples,).
        target_class (int, optional): The class to oversample. Default is 2.

    Returns:
        tuple:
            - ndarray: Augmented feature matrix with new synthetic samples.
            - ndarray: Corresponding label vector including labels for new samples.
    """
    needs_aug, target_count, _, _, _ = get_gap_class_target_count(
        Y, target_class, ratio=gap_ratio
    )
    if not needs_aug:
        print("No SVM-SMOTE needed.")
        return X, Y

    try:
        smoter = SVMSMOTE(
            sampling_strategy={target_class: target_count},
            random_state=config["random_state"],
        )
        X_aug, Y_aug = smoter.fit_resample(X, Y)
    except ValueError as e:
        print(f"SVMSMOTE failed: {e}")
        return X, Y

    return X_aug, Y_aug


def augment_borderline_smote_gap_class(
    X, Y, target_class=config["gap_class_label"], gap_ratio=config["gap_ratio"]
):
    needs_aug, target_count, _, _, _ = get_gap_class_target_count(
        Y, target_class, ratio=gap_ratio
    )
    if not needs_aug:
        print("No Borderline-SMOTE needed.")
        return X, Y

    try:
        smoter = BorderlineSMOTE(
            sampling_strategy={target_class: target_count},
            random_state=config["random_state"],
        )
        X_aug, Y_aug = smoter.fit_resample(X, Y)
    except ValueError as e:
        print(f"BorderlineSMOTE failed: {e}")
        return X, Y

    return X_aug, Y_aug


def augment_adasyn_gap_class(
    X, Y, target_class=config["gap_class_label"], gap_ratio=config["gap_ratio"]
):
    needs_aug, target_count, _, _, _ = get_gap_class_target_count(
        Y, target_class, ratio=gap_ratio
    )
    if not needs_aug:
        print("No ADASYN needed.")
        return X, Y

    try:
        adasyn = ADASYN(
            sampling_strategy={target_class: target_count},
            random_state=config["random_state"],
        )
        X_aug, Y_aug = adasyn.fit_resample(X, Y)
    except ValueError as e:
        print(f"ADASYN failed: {e}")
        return X, Y

    return X_aug, Y_aug


def get_gap_class_target_count(
    Y, target_class=config["gap_class_label"], ratio=config["gap_ratio"]
):
    """
    Computes the target count for the gap class (label 2) and returns
    whether augmentation is needed based on the current distribution.

    Args:
        Y (ndarray): Label vector.
        target_class (int): The label for the gap class (default: 2).

    Returns:
        tuple:
            - bool: Whether augmentation is needed.
            - int: Target number of samples for the gap class.
    """
    class_counts = {
        label: np.sum(Y == label) for label in np.unique(Y) if label != target_class
    }

    if len(class_counts) < 2:
        # Not enough base classes to compute average of top two
        return False, 0

    # Get the counts of the two largest classes
    top_two_counts = sorted(class_counts.values(), reverse=True)[:2]
    avg_top_two = np.mean(top_two_counts)

    top_two_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    avg_top_two = np.mean([v for _, v in top_two_classes])
    top_two_labels = [k for k, _ in top_two_classes]

    target_count = int(ratio * avg_top_two)
    current_gap_count = np.sum(Y == target_class)

    needs_augmentation = current_gap_count < target_count
    return (
        needs_augmentation,
        target_count,
        avg_top_two,
        current_gap_count,
        top_two_labels,
    )
