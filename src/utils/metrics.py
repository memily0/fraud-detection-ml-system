from sklearn.metrics import (
    auc,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def pr_auc_score(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return auc(recall, precision)


def roc_auc(y_true, y_proba):
    return roc_auc_score(y_true, y_proba)


def f2_score(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba > threshold).astype(int)
    return fbeta_score(y_true, y_pred, beta=2)


def precision_at_threshold(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba > threshold).astype(int)
    return precision_score(y_true, y_pred, zero_division=0)


def recall_at_threshold(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba > threshold).astype(int)
    return recall_score(y_true, y_pred, zero_division=0)
