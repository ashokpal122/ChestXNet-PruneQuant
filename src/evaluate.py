import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from dataloader import make_dataloaders
from model import build_resnet50_head

def evaluate(model_path, data_dir, batch_size=32, img_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, classes = make_dataloaders(data_dir, batch_size=batch_size, img_size=img_size)
    model = build_resnet50_head(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device).eval()

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs.cpu(), dim=1).numpy()
            preds = probs.argmax(axis=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_probs.extend(probs.tolist())

    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='viridis')
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()
    evaluate(args.model_path, args.data_dir, args.batch_size, args.img_size)