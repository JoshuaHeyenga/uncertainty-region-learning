from dataset import generate_dataset, split_dataset
from model import (
    assign_gap_class,
    augment_gap_class,
    clean_train_classifier,
    evaluate_model,
)
from visualization import plot_results_with_decision_boundary


def main():
    """
    Main function to execute the workflow.
    """
    # Base Setup
    X, Y = generate_dataset()
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)
    classifier = clean_train_classifier(X_train, Y_train)
    evaluate_model(classifier, X_test, Y_test)
    plot_results_with_decision_boundary(classifier, X, Y)

    Y_with_gap, gap_mask = assign_gap_class(classifier, X, Y)

    # Now oversample ONLY class 2 within the full dataset
    X_aug, Y_aug = augment_gap_class(X, Y_with_gap)

    classifier_aug = clean_train_classifier(X_aug, Y_aug)

    # Now plot the augmented version
    evaluate_model(classifier_aug, X_test, Y_test)
    plot_results_with_decision_boundary(classifier_aug, X, Y_with_gap)


if __name__ == "__main__":
    main()
