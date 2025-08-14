import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from config import CONFIG

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
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    GAP = CONFIG["first_gap_class_label"]

    # 1) Classes in the exact model order
    classes = classifier.classes_.astype(int)  # e.g. [0,1,2,3,104]
    is_gap = classes == GAP
    base_classes = classes[~is_gap]  # exclude gap from surface

    # 2) Colors/labels (stable order = base_classes, then gap for points)
    base_cmap = plt.get_cmap("tab10")
    class_colors = {int(c): base_cmap(i) for i, c in enumerate(base_classes)}
    gap_color = "#FFB000"
    if GAP in np.unique(Y):
        class_colors[int(GAP)] = gap_color

    class_labels = {int(c): f"Class {int(c)}" for c in base_classes}
    class_labels[int(GAP)] = "Gap Class"

    # 3) Grid + probabilities (exclude gap column for surface)
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    P = classifier.predict_proba(grid)  # shape (N, K) in classes_ order
    P_base = P[:, ~is_gap]  # drop gap column
    Z_pos = np.argmax(P_base, axis=1).reshape(xx.shape)  # positions 0..len(base)-1

    if ax is None:
        _, ax = plt.subplots()

    # 4) Surface with colormap that matches base_classes order
    cmap = ListedColormap([class_colors[int(c)] for c in base_classes])
    ax.contourf(xx, yy, Z_pos, cmap=cmap, alpha=0.3)

    # 5) Scatter original classes
    for cls in base_classes:
        cls = int(cls)
        ax.scatter(
            X[Y == cls, 0],
            X[Y == cls, 1],
            c=[class_colors[cls]],
            edgecolor="k",
            label=class_labels[cls],
            s=20,
        )

    # Gap points (points only, no surface)
    if GAP in np.unique(Y):
        ax.scatter(
            X[Y == GAP, 0],
            X[Y == GAP, 1],
            c=[gap_color],
            edgecolor="k",
            label=class_labels[GAP],
            s=20,
        )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()


def plot_performance_across_ratios(file_path: str, obs_class: int):
    total_df = pd.read_csv(file_path)

    # Filter for the observed class
    df = total_df[total_df["class"] == obs_class]

    # Metrics to include
    metrics = ["precision", "recall", "f1", "accuracy"]

    # Group: average + std across seeds and thresholds
    grouped_mean = (
        df.groupby(["stage", "gap_ratio"])[metrics]
        .mean()
        .reset_index()
        .sort_values("gap_ratio")
    )

    grouped_std = (
        df.groupby(["stage", "gap_ratio"])[metrics]
        .std()
        .reset_index()
        .sort_values("gap_ratio")
    )

    # Separate pre and post stages
    pre_mean_df = grouped_mean[grouped_mean["stage"] == "pre"]
    post_mean_df = grouped_mean[grouped_mean["stage"] == "post"]
    post_std_df = grouped_std[grouped_std["stage"] == "post"]

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(7, 4))

    for metric in metrics:
        color = COLORS[metric]

        # === Post values with std fill ===
        ax.plot(
            post_mean_df["gap_ratio"],
            post_mean_df[metric],
            label=f"Post {metric.capitalize()}",
            marker="o",
            color=color,
        )

        ax.fill_between(
            post_mean_df["gap_ratio"],
            post_mean_df[metric] - post_std_df[metric],
            post_mean_df[metric] + post_std_df[metric],
            alpha=0.15,
            color=color,
        )

        # === Pre baseline (dashed horizontal line) ===
        ax.axhline(
            y=pre_mean_df[metric].mean(),  # mean over all gap ratios
            linestyle="--",
            color=color,
        )

    # === Legend ===
    pre_legend_proxy = Line2D(
        [0], [0], linestyle="--", color="gray", label="Pre values"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, pre_legend_proxy)
    labels.insert(0, "Pre values")

    if obs_class == 0:
        ax.legend(
            handles,
            labels,
            title="Metric & Stage",
            loc="lower right",
            fontsize=13,
            title_fontsize=14,
        )

    # === Axes & Labels ===
    ax.set_xlabel("Gap Ratio", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True)
    ax.tick_params(axis="both", labelsize=13)

    plt.tight_layout()
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
        )

    ax.set_xlabel("Gap Ratio", fontsize=16)
    ax.set_ylabel(metric.capitalize(), fontsize=16)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True)
    if obs_class == 0:
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

    metrics = ["precision", "recall", "f1", "accuracy"]

    # Separate pre and post stages
    pre_df = method_df[method_df["stage"] == "pre"]
    post_df = method_df[method_df["stage"] == "post"]

    # Group by threshold: compute mean and std
    pre_grouped = (
        pre_df.groupby("threshold")[metrics]
        .mean()
        .reset_index()
        .sort_values("threshold")
    )
    post_grouped_mean = (
        post_df.groupby("threshold")[metrics]
        .mean()
        .reset_index()
        .sort_values("threshold")
    )
    post_grouped_std = (
        post_df.groupby("threshold")[metrics]
        .std()
        .reset_index()
        .sort_values("threshold")
    )

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in metrics:
        color = COLORS.get(metric, None)

        # Pre-gap (dashed line)
        ax.plot(
            pre_grouped["threshold"],
            pre_grouped[metric],
            linestyle="--",
            color=color,
        )

        # Post-gap (solid line)
        ax.plot(
            post_grouped_mean["threshold"],
            post_grouped_mean[metric],
            marker="o",
            linestyle="-",
            label=f"Post {metric.capitalize()}",
            color=color,
        )

        # Std deviation fill
        ax.fill_between(
            post_grouped_mean["threshold"],
            post_grouped_mean[metric] - post_grouped_std[metric],
            post_grouped_mean[metric] + post_grouped_std[metric],
            alpha=0.15,
            color=color,
        )

    # === Legend ===
    pre_legend_proxy = Line2D(
        [0], [0], linestyle="--", color="gray", label="Pre values"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, pre_legend_proxy)
    labels.insert(0, "Pre values")
    if gap_ratio == 0.05:
        ax.legend(
            handles,
            labels,
            title="Metric & Stage",
            loc="lower right",
            fontsize=13,
            title_fontsize=14,
        )

    # === Axes & Labels ===
    ax.set_xlabel("Threshold", fontsize=16)
    ax.set_ylabel("Score", fontsize=16)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True)
    ax.tick_params(axis="both", labelsize=13)

    plt.tight_layout()
    plt.show()


def plot_gcg(file_path: str, augment_method: str, gap_ratio: float):
    total_df = pd.read_csv(file_path)

    gcg_df = total_df[
        (total_df["method"] == augment_method)
        & (total_df["stage"] == "post")
        & (total_df["gap_ratio"] == gap_ratio)
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


def plot_gcg_across_ratios(file_path: str, augment_method: str):
    total_df = pd.read_csv(file_path)

    # Filter for the relevant method and post-augmentation stage
    gcg_df = total_df[
        (total_df["method"] == augment_method) & (total_df["stage"] == "post")
    ]

    # Define the specific gap ratios to plot
    target_ratios = [0.05, 0.1, 0.25, 0.5]

    fig, ax = plt.subplots(figsize=(7, 4))

    for ratio in target_ratios:
        ratio_df = gcg_df[gcg_df["gap_ratio"] == ratio]

        grouped_mean = (
            ratio_df.groupby("threshold")["gcg"]
            .mean()
            .reset_index()
            .sort_values("threshold")
        )

        grouped_std = (
            ratio_df.groupby("threshold")["gcg"]
            .std()
            .reset_index()
            .sort_values("threshold")
        )

        # Plot mean line
        ax.plot(
            grouped_mean["threshold"],
            grouped_mean["gcg"],
            marker="o",
            label=f"Gap Ratio {ratio:.2f}",
        )

        # Plot std band
        ax.fill_between(
            grouped_mean["threshold"],
            grouped_mean["gcg"] - grouped_std["gcg"],
            grouped_mean["gcg"] + grouped_std["gcg"],
            alpha=0.15,
        )

    ax.set_xlabel("Threshold", fontsize=16)
    ax.set_ylabel("GCG", fontsize=16)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True)
    ax.legend(
        loc="lower right",
        fontsize=12,
        title="Gap Ratio",
        title_fontsize=13,
    )
    ax.tick_params(axis="both", labelsize=13)

    plt.tight_layout()
    plt.show()
