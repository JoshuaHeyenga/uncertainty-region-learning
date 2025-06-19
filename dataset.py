import yaml
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def generate_dataset():
    """
    Generates a synthetic dataset using Gaussian blobs.

    The number of samples, centers (clusters), standard deviation, and random state
    are defined in the external configuration file `config.yaml`.

    Returns:
        X (ndarray): Generated feature matrix of shape (n_samples, n_features).
        Y (ndarray): Corresponding label vector of shape (n_samples,).
    """

    mode = "make_blobs"

    if mode == "make_blobs":
        X, Y = make_blobs(
            n_samples=config["n_samples"],
            centers=config["centers"],
            cluster_std=config["cluster_std"],
            random_state=config["random_state"],
        )
        return X, Y

    if mode == "make_classification":
        X, Y = make_classification(
            n_samples=config["n_samples"],
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=0.8,  # < 1 = more overlap
            flip_y=0.01,  # 1% label noise
            weights=[0.6, 0.4],  # optional imbalance
            random_state=config["random_state"],
        )
        return X, Y


def split_dataset(X, Y, test_size=0.2):
    """
    Splits a dataset into training and test subsets.

    Uses a fixed random seed from the configuration for reproducibility.

    Args:
        X (ndarray): Full feature matrix of shape (n_samples, n_features).
        Y (ndarray): Full label vector of shape (n_samples,).
        test_size (float): Fraction of the dataset to use for the test set (default: 0.2).

    Returns:
        tuple:
            - X_train (ndarray): Feature matrix for the training set.
            - X_test (ndarray): Feature matrix for the test set.
            - Y_train (ndarray): Label vector for the training set.
            - Y_test (ndarray): Label vector for the test set.
    """

    return train_test_split(
        X, Y, test_size=test_size, random_state=config["random_state"]
    )
