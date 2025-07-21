from enum import Enum


class AugmentationMethod(Enum):
    """
    Enum for different augmentation methods.
    """

    SMOTE = "smote"
    OVERSAMPLING = "oversampling"
    SVM_SMOTE = "svm_smote"
    BORDERLINE_SMOTE = "borderline_smote"
    ADASYN = "adasyn"


class PerformanceMetric(Enum):
    """
    Enum for the different performance metrics.
    """

    PRECISION = "precision"
    ACCURACY = "accuracy"
    F1SCORE = "f1"
    RECALL = "recall"


class PerformanceStage(Enum):
    PRE = "pre"
    POST = "post"


class Dataset(Enum):
    """
    Enum for different datasets used in the experiments.
    """

    BLOBS = "blobs"
    MULTI_BLOBS = "multi_blobs"
    IRIS = "iris"
    WINE = "wine"
    CANCER = "cancer"
