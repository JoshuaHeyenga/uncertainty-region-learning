import pandas as pd
import yaml
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from enums import Dataset

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def generate_dataset(mode: Dataset) -> tuple:
    """
    Generates and returns feature and label data for one of the supported datasets.

    Depending on the `mode`, the dataset is generated synthetically or loaded from a file.

    Returns:
        tuple:
            - X (ndarray): Feature matrix of shape (n_samples, n_features).
            - Y (ndarray): Label vector of shape (n_samples,).
    """

    if mode == Dataset.BLOBS:
        X, Y = make_blobs(
            n_samples=config["n_samples"],
            centers=config["centers"],
            cluster_std=config["cluster_std"],
            random_state=config["random_state"],
        )
        return X, Y

    if mode == Dataset.MULTI_BLOBS:
        X, Y = make_blobs(
            n_samples=2000,
            centers=4,
            cluster_std=5,
            random_state=config["random_state"],
        )
        return X, Y

    if mode == Dataset.WINE:
        df = pd.read_csv("data/WineQT.csv")

        df = df.dropna()
        Y = df["quality"].values
        X = df.drop(columns=["quality", "Id"]).values

        X = StandardScaler().fit_transform(X)

        return X, Y

    if mode == Dataset.CANCER:
        df = pd.read_csv("data/wdbc.data", header=None)

        columns = ["ID", "diagnosis"] + [f"feature_{i}" for i in range(30)]
        df.columns = columns

        df = df.drop(columns=["ID"])
        df = df.dropna()

        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

        X = df.drop(columns=["diagnosis"]).values
        Y = df["diagnosis"].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, Y


def split_dataset(X, Y, test_size=0.5):
    """
    Splits a dataset into training and test subsets.

    Uses a fixed random seed from the configuration for reproducibility.

    Args:
        X (ndarray): Full feature matrix of shape (n_samples, n_features).
        Y (ndarray): Full label vector of shape (n_samples,).
        test_size (float): Fraction of the dataset to use for the test set (default: 0.5).

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
