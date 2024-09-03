from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

def hter_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        raise ValueError("Confusion matrix must be 2x2 for binary classification.")

    FAR = fp / (fp + tn) if (fp + tn) > 0 else 0
    FRR = fn / (fn + tp) if (fn + tp) > 0 else 0
    HTER = (FAR + FRR) / 2
    
    return HTER



def evaluate_model(y_pred_prob_train, y_train, y_pred_prob_val, y_val, threshold = 0.5):

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
    return HTER_train, HTER_val, f1_train, f1_val, roc_auc_train, roc_auc_val 
    

