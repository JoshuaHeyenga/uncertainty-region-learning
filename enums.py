from enum import Enum


class AugmentationMethod(Enum):
    """
    Enum for different augmentation methods.
    """

    SMOTE = "smote"
    OVERSAMPLING = "oversampling"
    SVM_SMOTE = "svm_smote"


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
