import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

GAP_LABEL = config["gap_class_label"]


def plot_results_with_decision_boundary(
    classifier, X, Y, ax=None, title="", mode="auto"
):
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

    unique_classes = np.unique(Y)
    base_classes = [c for c in unique_classes if c != GAP_LABEL]

    # Color palette for base classes
    base_cmap = plt.get_cmap("tab10")
    class_colors = {cls: base_cmap(i) for i, cls in enumerate(base_classes)}
    class_labels = {cls: f"Class {cls}" for cls in base_classes}

    # Add gap class color
    if GAP_LABEL in unique_classes:
        class_colors[GAP_LABEL] = "#FFA500"
        class_labels[GAP_LABEL] = "Gap Class"

    # Create grid for decision surface
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.argmax(classifier.predict_proba(grid), axis=1)
    Z = Z.reshape(xx.shape)

    if ax is None:
        fig, ax = plt.subplots()

    # Plot decision surface
    cmap = ListedColormap([class_colors[c] for c in sorted(class_colors)])
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.3)

    # Plot data points
    for cls in sorted(class_colors.keys()):
        ax.scatter(
            X[Y == cls, 0],
            X[Y == cls, 1],
            c=[class_colors[cls]],
            edgecolor="k",
            label=class_labels[cls],
            s=20,
        )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
