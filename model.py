import numpy as np
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def clean_train_classifier(X_train, Y_train):
    """
    Trains a classifier on the training data.

    Args:
        X_train (ndarray): Training feature matrix.
        Y_train (ndarray): Training label vector.

    Returns:
        classifier (MLPClassifier): Trained classifier.
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
    bla
    """

    Y_pred = classifier.predict(X_test)
    print(
        "Classification Report:\n",
        classification_report(Y_test, Y_pred, labels=[0, 1], zero_division=0),
    )


def assign_gap_class(classifier, X, Y):
    proba = classifier.predict_proba(X)
    confidence = np.max(proba, axis=1)

    uncertain_mask = confidence < (1 - config["uncertainty_threshold"])

    Y_extended = np.copy(Y)
    Y_extended[uncertain_mask] = 2
    return Y_extended, uncertain_mask


def augment_smote_gap_class(X, Y, target_class=2):
    n_gap = np.sum(Y == target_class)
    desired_total = 2 * n_gap

    if n_gap >= desired_total:
        print("No augmentation needed.")
        return X, Y

    smoter = SMOTE(sampling_strategy={target_class: desired_total})
    X_aug, Y_aug = smoter.fit_resample(X, Y)

    return X_aug, Y_aug
