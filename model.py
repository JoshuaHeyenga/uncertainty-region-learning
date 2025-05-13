import numpy as np
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

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


def evaluate_model(classifier, X_test, Y_test):
    """
    Evaluates the performance of the trained classifier on test data and prints a classification report.

    The report includes precision, recall, F1-score, and support for classes 0 and 1.

    Args:
        classifier (MLPClassifier): The trained classifier.
        X_test (ndarray): Feature matrix for evaluation, shape (n_samples, n_features).
        Y_test (ndarray): True labels for the test set, shape (n_samples,).
    """

    Y_pred = classifier.predict(X_test)
    print(
        "Classification Report:\n",
        classification_report(Y_test, Y_pred, labels=[0, 1], zero_division=0),
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
    desired_total = 2 * n_gap

    if n_gap >= desired_total:
        print("No augmentation needed.")
        return X, Y

    smoter = SMOTE(sampling_strategy={target_class: desired_total})
    X_aug, Y_aug = smoter.fit_resample(X, Y)

    return X_aug, Y_aug
