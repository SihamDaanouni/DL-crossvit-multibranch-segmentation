import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def compute_classif_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="binary"))
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}