import numpy as np
import yaml
from imblearn.over_sampling import SMOTE
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
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
    )
    classifier.fit(X_train, Y_train)
    return classifier


def evaluate_and_log_model(
    classifier, X_test, Y_test, file_path, method, stage, seed, threshold
):
    Y_pred = classifier.predict(X_test)

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        Y_test, Y_pred, labels=[0, 1], zero_division=0
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
        )


def assign_gap_class(classifier, X, Y):
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

    proba = classifier.predict_proba(X)
    confidence = np.max(proba, axis=1)

    uncertain_mask = confidence < (1 - config["uncertainty_threshold"])

    Y_extended = np.copy(Y)
    Y_extended[uncertain_mask] = 2
    return Y_extended, uncertain_mask


def augment_oversampling_gap_class(X, Y, target_class=2):
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

    X_target = X[Y == target_class]
    Y_target = Y[Y == target_class]

    # Compute number of samples to generate
    n_existing = len(X_target)
    n_desired = 2 * n_existing  # Check if the factor could be a variable

    if n_existing >= n_desired:
        print("No oversampling needed.")
        return X, Y

    # Resample with replacement
    X_oversampled, Y_oversampled = resample(
        X_target,
        Y_target,
        replace=True,
        n_samples=n_desired - n_existing,
        random_state=42,
    )

    # Concatenate original and new samples
    X_augmented = np.vstack((X, X_oversampled))
    Y_augmented = np.hstack((Y, Y_oversampled))

    return X_augmented, Y_augmented


def augment_smote_gap_class(X, Y, target_class=2):
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

    n_gap = np.sum(Y == target_class)
    desired_total = 5 * n_gap

    if n_gap >= desired_total:
        print("No augmentation needed.")
        return X, Y

    smoter = SMOTE(sampling_strategy={target_class: desired_total})
    X_aug, Y_aug = smoter.fit_resample(X, Y)

    return X_aug, Y_aug
