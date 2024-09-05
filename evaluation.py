from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import numpy as np


def hter_metrics(y_true:np.ndarray , y_pred: np.ndarray):
    """
    Implement HTER (Half Total Error Rate)metrics.
    HTER is half of the sum of the false acceptance rate (FAR) and the false rejection rate (FRR)
    HTER is a performance metric used to evaluate binary classification systems, 
    particularly in the context of biometric systems. [1]
    It is calculated as the average of the False Acceptance Rate (FAR) and the False Rejection Rate (FRR).

    FPR = FP/(TN+FP) (x-axis in ROC curve, sensitivity max when FPR min) 
    TPR = FN/(TP+FN) (x-axis in ROC curve, specificity max when TPR min)

    [1] Caixun Wang, Jie Zhou,
    An adaptive index smoothing loss for face anti-spoofing,
    Pattern Recognition Letters,
    Volume 153, 2022, Pages 168-175, ISSN 0167-8655,
    https://doi.org/10.1016/j.patrec.2021.12.006.

    Parameters:
        y_true (np.ndarray): A NumPy array of y ground truth labels.
        y_pred (np.ndarray): A NumPy array of y prediction.

        
    Returns:
        float: result of the metric HTER.

        
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        raise ValueError("Confusion matrix must be 2x2 for binary classification.")

    FAR = fp / (fp + tn) if (fp + tn) > 0 else 0
    FRR = fn / (fn + tp) if (fn + tp) > 0 else 0
    HTER = (FAR + FRR) / 2
    
    return HTER



def evaluate_model(y_pred_prob_train:np.ndarray, y_train:np.ndarray, y_pred_prob_val:np.ndarray, y_val:np.ndarray, threshold = 0.5):
    """
    Evaluation of the model on f1_score, roc_auc_score and HTER score with threshold 0.5.
    The threshold is fine-tuned based on HTER in a second phase once the model is trained.

    Parameters:
        y_pred_prob_train (np.ndarray): A NumPy array of rediction [0,1] from Training set.
        y_train (np.ndarray): A NumPy array of y ground truth labels from Training set.
        y_pred_prob_val (np.ndarray): A NumPy array of rediction [0,1] from Validation set.
        y_val(np.ndarray): A NumPy array of y ground truth labels from Validation set.

        
    Returns:
        list of float: result of HTER_train, HTER_val, f1_train, f1_val, roc_auc_train, roc_auc_val 
    """
    y_pred_train = (y_pred_prob_train > threshold).astype(int)
    y_pred_val = (y_pred_prob_val > threshold).astype(int)

    f1_train = f1_score(y_train, y_pred_train)
    roc_auc_train = roc_auc_score(y_train, y_pred_train)

    f1_val = f1_score(y_val, y_pred_val)
    roc_auc_val = roc_auc_score(y_val, y_pred_val)

    HTER_train = hter_metrics(y_train, y_pred_train)
    HTER_val = hter_metrics(y_val, y_pred_val)

    print(f"Train F1 Score: {f1_train:.4f}")
    print(f"Validation F1 Score: {f1_val:.4f}")
    print(f"Train ROC AUC: {roc_auc_train:.4f}")
    print(f"Validation ROC AUC: {roc_auc_val:.4f}")
    print(f'HTER Train: {HTER_train:.4f}')
    print(f'HTER Val: {HTER_val:.4f}')
    print('\n')
    return HTER_train, HTER_val, f1_train, f1_val, roc_auc_train, roc_auc_val 
    

