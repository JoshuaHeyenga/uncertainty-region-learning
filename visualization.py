import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_results_with_decision_boundary(classifier, X, Y, ax=None, title=""):
    """
    Visualizes the decision boundary of a binary classifier (class 0 vs class 1)
    along with the data points, while excluding class 2 from decision
    boundary computation.

    This function manually computes the decision surface using the classifier's
    predicted probabilities and only uses the first two classes (0 and 1) for
    visualization. Points from all classes are plotted, including the gap class
    (shown in orange), but the decision boundary is only between classes 0 and 1.

    Args:
        classifier: A trained classifier with a `predict_proba` method (e.g. MLPClassifier).
        X (ndarray): Feature matrix of shape (n_samples, 2). Assumes two-dimensional input for plotting.
        Y (ndarray): Label vector of shape (n_samples,) with class labels 0, 1, and optionally 2.
        ax (matplotlib.axes.Axes, optional): Existing matplotlib axis to plot on. If None, a new figure is created.
        title (str, optional): Title to set for the plot.

    Notes:
        - Class 0 points are shown in red.
        - Class 1 points are shown in blue.
        - Class 2 (gap) points are shown in orange.
        - The decision boundary only separates class 0 and 1 regions.
    """

    class_colors = {0: "#FF0000", 1: "#0000FF", 2: "#FFA500"}
    bg_colors = ["#FFAAAA", "#AAAAFF"]

    # Generate grid
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    proba = classifier.predict_proba(grid)
    Z = np.argmax(proba[:, :2], axis=1).reshape(xx.shape)

    # Create axis if not passed
    if ax is None:
        fig, ax = plt.subplots()

    # Plot decision surface and data points
    ax.contourf(xx, yy, Z, cmap=ListedColormap(bg_colors), alpha=0.5)

    # Plot actual class 0 and 1 points
    class_labels = ["Class 0", "Class 1", "Gap Class"]
    for class_value in np.unique(Y):
        ax.scatter(
            X[Y == class_value, 0],
            X[Y == class_value, 1],
            c=class_colors[class_value],
            edgecolor="k",
            s=20,
            label=class_labels[class_value],
        )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
