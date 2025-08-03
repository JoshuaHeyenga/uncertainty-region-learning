import pandas as pd
import yaml
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from enums import Dataset

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def generate_dataset(mode: Dataset) -> tuple:
    """
    Generates a synthetic dataset using Gaussian blobs.

    The number of samples, centers (clusters), standard deviation, and random state
    are defined in the external configuration file `config.yaml`.

    Returns:
        X (ndarray): Generated feature matrix of shape (n_samples, n_features).
        Y (ndarray): Corresponding label vector of shape (n_samples,).
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

    if mode == Dataset.IRIS:
        df = pd.read_csv("data/Iris.csv")

        # Drop Id column if exists
        df = df.drop(columns=["Id"], errors="ignore")
        # df = df[df["Species"].isin(["Iris-setosa", "Iris-versicolor"])]

        # Encode species as integer labels (0, 1, 2)
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(df["Species"])
        X = df.drop(columns=["Species"]).values

        # Optional: normalize features
        X = StandardScaler().fit_transform(X)

        return X, Y

    if mode == Dataset.WINE:
        df = pd.read_csv("data/WineQT.csv")

        # Drop any missing values just in case
        df = df.dropna()

        # if binary:
        # Binary classification: Good (6 or higher) vs. Bad
        #    df["quality"] = (df["quality"] >= 6).astype(int)

        Y = df["quality"].values
        X = df.drop(columns=["quality"]).values

        # Normalize features
        X = StandardScaler().fit_transform(X)

        return X, Y

    if mode == Dataset.CANCER:
        df = pd.read_csv("data/wdbc.data", header=None)

        columns = ["ID", "diagnosis"] + [f"feature_{i}" for i in range(30)]
        df.columns = columns

        df = df.drop(columns=["ID"])

        df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

        X = df.drop(columns=["diagnosis"]).values
        Y = df["diagnosis"].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

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
