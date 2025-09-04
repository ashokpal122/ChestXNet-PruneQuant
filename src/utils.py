import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, roc_auc_score, average_precision_score)
from sklearn.preprocessing import label_binarize

def compute_epoch_metrics(y_true, y_probs, y_pred, n_classes):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_macro'] = prec
    metrics['recall_macro'] = rec
    metrics['f1_macro'] = f1
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        metrics['auc'] = roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr')
        metrics['auc_pr'] = average_precision_score(y_true_bin, y_probs, average='macro')
    except Exception:
        metrics['auc'] = np.nan
        metrics['auc_pr'] = np.nan
    return metrics

def plot_training_curves(history, save_path=None):
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1); plt.plot(epochs, history['train_loss'], label='train'); plt.plot(epochs, history['val_loss'], label='val'); plt.title('Loss'); plt.legend()
    plt.subplot(2,2,2); plt.plot(epochs, history['train_acc'], label='train'); plt.plot(epochs, history['val_acc'], label='val'); plt.title('Accuracy'); plt.legend()
    plt.subplot(2,2,3); plt.plot(epochs, history['train_precision'], label='train'); plt.plot(epochs, history['val_precision'], label='val'); plt.title('Precision'); plt.legend()
    plt.subplot(2,2,4); plt.plot(epochs, history['train_recall'], label='train'); plt.plot(epochs, history['val_recall'], label='val'); plt.title('Recall'); plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)