from enum import Enum


class AugmentationMethod(Enum):
    SMOTE = "smote"
    OVERSAMPLING = "oversampling"
    SVM_SMOTE = "svm_smote"
    BORDERLINE_SMOTE = "borderline_smote"
    ADASYN = "adasyn"


class PerformanceMetric(Enum):
    PRECISION = "precision"
    ACCURACY = "accuracy"
    F1SCORE = "f1"
    RECALL = "recall"


class PerformanceStage(Enum):
    PRE = "pre"
    POST = "post"


class Dataset(Enum):
    BLOBS = "blobs"
    MULTI_BLOBS = "multi_blobs"
    WINE = "wine"
    CANCER = "cancer"
