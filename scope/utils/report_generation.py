import numpy as np
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score,
                             confusion_matrix, f1_score, log_loss)


def make_report(y_true: np.ndarray, y_pred: np.ndarray, y_pred_softmax: np.ndarray) -> dict:
    """
    Generates a comprehensive performance report for results, including 
    ROC-related metrics, F1 scores, log loss, and confusion matrix evaluation.

    The function computes the receiver operating characteristic (ROC) metrics,
    such as false positive rates (fpr), true positive rates (tpr), thresholds,
    area under the ROC curve (auc_roc), and classification evaluation metrics such as
    accuracy score, F1 scores, log loss and confusion matrix from true labels and predicted values.

    Arguments:
        y_true: np.ndarray
            True binary class labels of the dataset (0 or 1).
        y_pred: np.ndarray
            Predicted binary class labels by a classifier (0 or 1).
        y_pred_softmax: np.ndarray
            Predicted probabilities for each class [prob_class_0, prob_class_1], 
            used for generating the ROC curve and calculating log loss.

    Returns:
        dict:
            A dictionary containing the computed performance metrics with the keys:
            - 'fpr': array of false positive rates.
            - 'tpr': array of true positive rates.
            - 'thresholds': array of decision thresholds for the ROC curve.
            - 'auc_roc': area under the ROC curve.
            - 'acc': accuracy score of the predictions.
            - 'f1_score': F1 score.
            - 'log_loss': logarithmic loss of the predictions.
            - 'confusion_matrix': confusion matrix of true vs predicted class labels.
    """
    
    y_pred_softmax_array = np.array(y_pred_softmax)
    
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_softmax_array[:, 1])
        auc_roc = roc_auc_score(y_true, y_pred_softmax_array[:, 1])
    except Exception as e:
        print(f"Warning: Error calculando ROC/AUC: {e}")
        fpr, tpr, thresholds = None, None, None
        auc_roc = 0.5
    
    try:
        acc = accuracy_score(y_true, y_pred)
    except Exception as e:
        print(f"Warning: Error calculando accuracy: {e}")
        acc = 0.0
    
    try:
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        print(f"Warning: Error calculando F1 score: {e}")
        f1 = 0.0
    
    try:
        logloss = log_loss(y_true, y_pred_softmax_array)
    except Exception as e:
        print(f"Warning: Error calculando log loss: {e}")
        logloss = 1.0
    
    try:
        conf_matrix = confusion_matrix(y_true, y_pred)
    except Exception as e:
        print(f"Warning: Error calculando confusion matrix: {e}")
        conf_matrix = None

    this_data: dict = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc_roc': auc_roc,
        'acc': acc,
        'f1_score': f1,
        'log_loss': logloss,
        'confusion_matrix': conf_matrix
    }

    return this_data