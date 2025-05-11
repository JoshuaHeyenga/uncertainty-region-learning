from dataset import generate_dataset, split_dataset
from model import clean_train_classifier, evaluate_model
from visualization import plot_results_with_decision_boundary


def main():
    """
    Main function to execute the workflow.
    """
    X, Y = generate_dataset()
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    classifier = clean_train_classifier(X_train, Y_train)
    evaluate_model(classifier, X_test, Y_test)

    plot_results_with_decision_boundary(classifier, X, Y)


if __name__ == "__main__":
    main()
