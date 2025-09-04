import os
import time
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import make_dataloaders
from model import build_resnet50_head
from utils import compute_epoch_metrics, plot_training_curves, save_model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, classes = make_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size, seed=args.seed
    )

    model = build_resnet50_head(num_classes=len(classes), pretrained=True).to(device)

    # compute class weights from train loader (simple approach)
    # (Alternatively compute in dataloader and pass back)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = defaultdict(list)
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        all_true, all_probs, all_preds = [], [], []

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (images, labels) in loop:
            images = images.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()

            running_loss += loss.item() * labels.size(0)
            probs = F.softmax(outputs.detach().cpu(), dim=1).numpy()
            preds = probs.argmax(axis=1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

        train_loss = running_loss / len(train_loader.dataset)
        train_metrics = compute_epoch_metrics(all_true, all_probs, all_preds, n_classes=len(classes))

        # Validation
        model.eval()
        val_loss = 0.0
        v_true, v_probs, v_preds = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device); labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                probs = F.softmax(outputs.cpu(), dim=1).numpy()
                preds = probs.argmax(axis=1)
                v_true.extend(labels.cpu().numpy().tolist())
                v_probs.extend(probs.tolist())
                v_preds.extend(preds.tolist())
        val_loss = val_loss / len(val_loader.dataset)
        val_metrics = compute_epoch_metrics(v_true, v_probs, v_preds, n_classes=len(classes))

        # log & save
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy']); history['val_acc'].append(val_metrics['accuracy'])
        history['train_precision'].append(train_metrics['precision_macro']); history['val_precision'].append(val_metrics['precision_macro'])
        history['train_recall'].append(train_metrics['recall_macro']); history['val_recall'].append(val_metrics['recall_macro'])

        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} - {int(epoch_time)}s - Train acc: {train_metrics['accuracy']:.4f} - Val acc: {val_metrics['accuracy']:.4f} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            save_model(model, best_path)
            print(f"Saved best model: {best_path}")

    # final save & plot
    final_path = os.path.join(args.save_dir, 'trained_model_final.pth')
    save_model(model, final_path)
    plot_training_curves(history, save_path=os.path.join(args.save_dir, 'training_curves.png'))
    print(f"Training finished. Final model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--save-dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)