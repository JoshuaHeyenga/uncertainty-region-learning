import numpy as np
import yaml
from imblearn.over_sampling import SMOTE, SVMSMOTE
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

from logger import log_metrics_to_csv

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def clean_train_classifier(X_train, Y_train):
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
        random_state=42,
    )
    classifier.fit(X_train, Y_train)
    return classifier


def evaluate_and_log_model(
    classifier, X_test, Y_test, file_path, method, stage, seed, threshold, gap_ratio
):
    Y_pred = classifier.predict(X_test)

    class_labels = sorted(list(set(np.unique(Y_test)) & set(np.unique(Y_pred))))
    class_labels = [
        label for label in class_labels if label != config["gap_class_label"]
    ]

    if not class_labels:
        print("No valid base classes to evaluate.")
        return

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        Y_test, Y_pred, labels=class_labels, zero_division=0
    )

    for class_label in [0, 1]:
        log_metrics_to_csv(
            file_path=file_path,
            method=method,
            stage=stage,
            seed=seed,
            class_label=class_label,
            threshold=threshold,
            precision=precision_arr[class_label],
            recall=recall_arr[class_label],
            f1=f1_arr[class_label],
            support=support_arr[class_label],
            gap_ratio=gap_ratio,
        )


def assign_gap_class(classifier, X, Y, threshold=config["uncertainty_threshold"]):
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

    proba = classifier.predict_proba(X)
    confidence = np.max(proba, axis=1)

    uncertain_mask = confidence < (1 - threshold)

    Y_extended = np.copy(Y)
    Y_extended[uncertain_mask] = config["gap_class_label"]
    return Y_extended, uncertain_mask


def augment_oversampling_gap_class(X, Y, target_class=config["gap_class_label"]):
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
    needs_aug, target_count = get_gap_class_target_count(Y, target_class)
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
        random_state=42,
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

    needs_aug, target_count, avg_top_two, current_gap_size = get_gap_class_target_count(
        Y, target_class, ratio=gap_ratio
    )

    print(f"--- SMOTE Decision Log ---")
    print(f"Target gap class label: {target_class}")
    print(f"Current gap class size: {current_gap_size}")
    print(f"Average of top 2 base classes: {avg_top_two}")
    print(f"Target count (gap_ratio={gap_ratio}): {target_count}")
    print(f"Needs augmentation: {needs_aug}")
    print(f"---------------------------")

    if not needs_aug:
        print("No SMOTE needed.")
        return X, Y
    else:
        print(f"Augmenting gap class {target_class} to target count: {target_count}")

    smoter = SMOTE(sampling_strategy={target_class: target_count}, random_state=42)
    X_aug, Y_aug = smoter.fit_resample(X, Y)

    return X_aug, Y_aug


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
    needs_aug, target_count = get_gap_class_target_count(
        Y, target_class, ratio=gap_ratio
    )
    if not needs_aug:
        print("No SVM-SMOTE needed.")
        return X, Y

    try:
        smoter = SVMSMOTE(
            sampling_strategy={target_class: target_count},
            random_state=42,
        )
        X_aug, Y_aug = smoter.fit_resample(X, Y)
    except ValueError as e:
        print(f"SVMSMOTE failed: {e}")
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

    target_count = int(ratio * avg_top_two)
    current_gap_count = np.sum(Y == target_class)

    needs_augmentation = current_gap_count < target_count
    return needs_augmentation, target_count, avg_top_two, current_gap_count
