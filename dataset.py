import yaml
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def generate_dataset():
    """
    Generates a synthetic dataset using Gaussian blobs.

    Returns:
        X (ndarray): Feature matrix.
        Y (ndarray): Label vector.
    """
    X, Y = make_blobs(
        n_samples=config["n_samples"],
        centers=config["centers"],
        cluster_std=config["cluster_std"],
        random_state=config["random_state"],
    )
    return X, Y


def split_dataset(X, Y, test_size=0.2):
    return train_test_split(
        X, Y, test_size=test_size, random_state=config["random_state"]
    )
