import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from config import CONFIG
from enums import AugmentationMethod

# === CONFIG ===

GAP_LABEL = CONFIG["first_gap_class_label"]
COLORS = {
    "precision": "gold",
    "recall": "crimson",
    "f1": "dodgerblue",
    "accuracy": "purple",
}


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


def plot_performance_accross_ratios(
    file_path: str, obs_class: int, n_cols: int, n_rows: int
):
    total_df = pd.read_csv(file_path)

    # == Data Setup ==
    # Metrics
    metrics: str = [
        "precision",
        "recall",
        "f1",
        "support",
        "accuracy",
    ]  # could be even more as the filter is further down
    mean_df = total_df.groupby(
        ["method", "stage", "class", "threshold", "gap_ratio"], as_index=False
    )[metrics].mean()

    try:
        df = mean_df[(mean_df["class"] == obs_class)]

    except Exception as e:
        print(f"Error in DataFrame filtering: {e}")

    CLEAN_THRESHOLDS = sorted(df["threshold"].dropna().unique())

    # == Plot Settings ==
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(5 * n_cols, 3 * n_rows),
        sharey=True,
        constrained_layout=True,
    )

    axes = axes.flatten()

    for ax, threshold in zip(axes, CLEAN_THRESHOLDS):
        sub_df = df[df["threshold"] == threshold]

        pre_df = mean_df[
            (mean_df["threshold"] == threshold)
            & (mean_df["class"] == obs_class)
            & (mean_df["stage"] == "pre")
        ]

        pre_mean_df = pre_df.groupby("gap_ratio").mean(numeric_only=True)

        post_mean_df = (
            sub_df[sub_df["stage"] == "post"]
            .groupby("gap_ratio")
            .mean(numeric_only=True)
        )

        # == Graph Visualization ==
        for metric in ["precision", "recall", "f1", "accuracy"]:
            ax.plot(
                post_mean_df.index,
                post_mean_df[metric],
                marker="o",
                label=f"Post {metric.capitalize()}",
                color=COLORS[metric],
            )
            ax.axhline(
                y=pre_mean_df[metric].mean(),
                linestyle="--",
                color=COLORS[metric],
            )

            ax.set_title(f"Threshold = {threshold}")
            ax.set_xlabel("Gap Ratio")
            ax.grid(True)
            if ax == axes[0]:
                ax.set_ylabel("Score")

                pre_legend_proxy = Line2D(
                    [0], [0], linestyle="--", color="gray", label="Pre values"
                )
                handles, labels = ax.get_legend_handles_labels()
                handles.insert(0, pre_legend_proxy)
                labels.insert(0, "Pre values")
                ax.legend(
                    handles,
                    labels,
                    title="Metric & Stage",
                    loc="lower right",
                )

    plt.show()


def plot_std_performance(
    file_path: str, obs_class: int, metric: str, stage: str, comp: bool
):
    total_df = pd.read_csv(file_path)

    total_df = total_df[(total_df["class"] == obs_class) & (total_df["stage"] == stage)]
    grouped_df = total_df.groupby(["gap_ratio"])[metric]
    mean = grouped_df.mean()
    std = grouped_df.std()

    # === Plot ===
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        mean.index,
        mean.loc[mean.index],
        label=f"{stage.capitalize()} {metric.capitalize()}",
        color="blue",  # make this dependent on metric
        marker="o",
    )

    ax.fill_between(
        mean.index,
        mean.loc[mean.index] - std.loc[mean.index],
        mean.loc[mean.index] + std.loc[mean.index],
        color="blue",  # make this dependent on metric
        alpha=0.2,
        label="± 1 Std. Dev.",
    )

    if comp:
        pre_df = pd.read_csv(file_path)
        pre_df = pre_df[(pre_df["class"] == obs_class) & (pre_df["stage"] == "pre")]
        pre_grouped_df = pre_df.groupby(["gap_ratio"])[metric]
        base_mean = pre_grouped_df.mean()

        ax.plot(
            base_mean.index,
            base_mean.loc[base_mean.index],
            label=f"Pre {metric.capitalize()}",
            color="red",
            marker="o",
            linestyle="--",
        )

        ax.fill_between(
            base_mean.index,
            base_mean.loc[base_mean.index] - std.loc[base_mean.index],
            base_mean.loc[base_mean.index] + std.loc[base_mean.index],
            color="red",  # make this dependent on metric
            alpha=0.2,
            label="± 1 Std. Dev.",  # how did i get +-
        )

    ax.set_xlabel("Gap Ratio", fontsize=16)
    ax.set_ylabel(metric.capitalize(), fontsize=16)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True)
    if obs_class == 0 and metric == "accuracy":
        ax.legend(loc="lower right", fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_performance_across_thresholds(
    file_path: str, augment_method: str, gap_ratio: float
):
    total_df = pd.read_csv(file_path)

    # Filter for method
    method_df = total_df[
        (total_df["method"] == augment_method) & (total_df["gap_ratio"] == gap_ratio)
    ]
    print(f"{len(method_df)} rows found for method '{augment_method}'")

    # Metrics to include
    metrics = ["precision", "recall", "f1", "accuracy"]

    # Separate pre and post stages
    pre_df = method_df[method_df["stage"] == "pre"]
    post_df = method_df[method_df["stage"] == "post"]

    # Group and average over thresholds
    pre_grouped = (
        pre_df.groupby("threshold")[metrics]
        .mean()
        .reset_index()
        .sort_values("threshold")
    )
    post_grouped = (
        post_df.groupby("threshold")[metrics]
        .mean()
        .reset_index()
        .sort_values("threshold")
    )

    # Plotting
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in metrics:
        # Pre-gap (dashed line)
        ax.plot(
            pre_grouped["threshold"],
            pre_grouped[metric],
            linestyle="--",
            color=COLORS.get(metric, None),
        )
        # Post-gap (solid line)
        ax.plot(
            post_grouped["threshold"],
            post_grouped[metric],
            marker="o",
            linestyle="-",
            label=f"Post {metric.capitalize()}",
            color=COLORS.get(metric, None),
        )

    # ax.set_title(f"Performance Across Thresholds ({augment_method})", fontsize=14)
    ax.set_xlabel("Threshold", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True)

    pre_legend_proxy = Line2D(
        [0], [0], linestyle="--", color="gray", label="Pre values"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, pre_legend_proxy)
    labels.insert(0, "Pre values")
    ax.legend(
        handles,
        labels,
        title="Metric & Stage",
        loc="lower right",
        fontsize=13,
        title_fontsize=14,
    )

    ax.tick_params(axis="both", labelsize=13)

    plt.tight_layout()
    plt.show()


def plot_gcg(file_path: str, augment_method: str):
    total_df = pd.read_csv(file_path)

    gcg_df = total_df[
        (total_df["method"] == augment_method) & (total_df["stage"] == "post")
    ]

    grouped = (
        gcg_df.groupby("threshold")["gcg"].mean().reset_index().sort_values("threshold")
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        grouped["threshold"],
        grouped["gcg"],
        marker="o",
        color="darkgreen",
        label="Avg GCG Score",
    )

    ax.set_xlabel("Threshold", fontsize=16)
    ax.set_ylabel("GCG", fontsize=16)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True)
    ax.legend(
        loc="lower right",
        fontsize=13,
        title_fontsize=14,
    )
    ax.tick_params(axis="both", labelsize=13)

    plt.tight_layout()
    plt.show()
