import numpy as np
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score,
                             confusion_matrix)


def make_report(y_true: np.ndarray, y_pred: np.ndarray, y_pred_auc: np.ndarray) -> dict:
    """
        Generates a performance report for classification results, including ROC-related
        metrics and confusion matrix evaluation.

        The function computes the receiver operating characteristic (ROC) metrics,
        such as false positive rates (fpr), true positive rates (tpr), thresholds,
        area under the ROC curve (auc_roc), and classification evaluation metrics such as
        accuracy score and confusion matrix from true labels and predicted values.

        Arguments:
            y_true: np.ndarray
                True class labels of the dataset.
            y_pred: np.ndarray
                Predicted class labels by a classifier.
            y_pred_auc: np.ndarray
                Predicted probabilities for the positive class, used for generating
                the ROC curve.

        Returns:
            dict:
                A dictionary containing the computed performance metrics with the keys:
                - 'fpr': array of false positive rates.
                - 'tpr': array of true positive rates.
                - 'thresholds': array of decision thresholds for the ROC curve.
                - 'auc_roc': area under the ROC curve.
                - 'acc': accuracy score of the predictions.
                - 'confusion_matrix': confusion matrix of true vs predicted class labels.
    """
    fpr, tpr, thresholds = roc_curve(y_true, np.array(y_pred_auc)[:, 1])
    auc_roc = roc_auc_score(y_true, np.array(y_pred_auc)[:, 1])

    this_data: dict = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc_roc': auc_roc,
        'acc': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    return this_data
