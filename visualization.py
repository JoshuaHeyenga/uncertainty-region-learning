import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ax = None
fig = None


def plot_results_with_decision_boundary(classifier, X, Y, *_):
    """
    Very basic visualization of dataset and decision boundary, following scikit-learn 1.17.4 example.
    """
    h = 0.02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    class_colors = {0: "#FF0000", 1: "#0000FF", 2: "#FFA500"}  # red, blue, orange

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading="auto")

    # Plot class 0, 1, and 2 with custom colors
    unique_classes = np.unique(Y)
    for class_value in unique_classes:
        color = class_colors.get(int(class_value), "black")
        plt.scatter(
            X[Y == class_value, 0],
            X[Y == class_value, 1],
            c=color,
            edgecolor="k",
            s=20,
            label=f"Class {int(class_value)}",
        )

    plt.title("Classifier Decision Boundary (Basic)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
