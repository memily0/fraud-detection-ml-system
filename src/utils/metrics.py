from sklearn.metrics import fbeta_score, precision_recall_curve, auc

def pr_auc_score(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)

def f2_score(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba > threshold).astype(int)
    return fbeta_score(y_true, y_pred, beta=2)