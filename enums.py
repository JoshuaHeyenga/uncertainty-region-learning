from enum import Enum


class AugmentationMethod(Enum):
    """
    Enum for different augmentation methods.
    """

    SMOTE = "smote"
    OVERSAMPLING = "oversampling"
    SVM_SMOTE = "svm_smote"
