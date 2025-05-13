import matplotlib.pyplot as plt

from dataset import generate_dataset, split_dataset
from model import (
    assign_gap_class,
    augment_smote_gap_class,
    clean_train_classifier,
    evaluate_model,
)
from visualization import plot_results_with_decision_boundary


def main():
    """
    Executes the full classification workflow including:

    1. Generating a synthetic dataset using Gaussian blobs.
    2. Training a baseline classifier on the original data.
    3. Evaluating and visualizing the decision boundary of the original model.
    4. Identifying uncertain data points using prediction confidence and assigning them to a 'gap' class (label 2).
    5. Augmenting the gap class using SMOTE to synthetically generate more representative samples.
    6. Retraining the classifier on the augmented dataset.
    7. Evaluating and visualizing the decision boundary after augmentation.

    Visual comparison of the decision boundary before and after gap-class augmentation is shown side-by-side.
    """

    X, Y = generate_dataset()
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
    classifier = clean_train_classifier(X_train, Y_train)
    evaluate_model(classifier, X_test, Y_test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_results_with_decision_boundary(classifier, X, Y, ax=axes[0], title="Pre-Gap")

    # Assign gap class and augment
    Y_with_gap, gap_mask = assign_gap_class(classifier, X, Y)
    X_aug, Y_aug = augment_smote_gap_class(X, Y_with_gap, target_class=2)
    classifier_aug = clean_train_classifier(X_aug, Y_aug)

    # Now plot the augmented version
    evaluate_model(classifier_aug, X_test, Y_test)
    plot_results_with_decision_boundary(
        classifier_aug, X_aug, Y_aug, ax=axes[1], title="Post-Gap"
    )  # Y_with_gap

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
