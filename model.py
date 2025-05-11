from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


def clean_train_classifier(X_train, Y_train):
    """
    bla
    """

    classifier = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )
    classifier.fit(X_train, Y_train)
    return classifier


def evaluate_model(classifier, X_test, Y_test):
    """
    bla
    """

    Y_pred = classifier.predict(X_test)
    print("Classification Report:\n", classification_report(Y_test, Y_pred))
