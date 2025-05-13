import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_results_with_decision_boundary(classifier, X, Y, ax=None, title=""):
    """
    Manually computes and plots decision boundary considering only class 0 and 1.
    Gap class (2) is excluded from the decision surface and label decisions.
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
    for class_value in np.unique(Y):
        ax.scatter(
            X[Y == class_value, 0],
            X[Y == class_value, 1],
            c=class_colors[class_value],
            edgecolor="k",
            s=20,
            label=f"Class {class_value}",
        )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
