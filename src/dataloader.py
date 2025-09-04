import os
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return train_transform, val_test_transform

def create_splits(data_dir, seed=42, splits=(0.7,0.15,0.15)):
    dataset = datasets.ImageFolder(root=data_dir)
    total = len(dataset)
    train_len = int(splits[0] * total)
    val_len = int(splits[1] * total)
    test_len = total - train_len - val_len

    indices = list(range(total))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_idx = indices[:train_len]
    val_idx = indices[train_len:train_len+val_len]
    test_idx = indices[train_len+val_len:]
    return dataset, train_idx, val_idx, test_idx

def make_dataloaders(data_dir, batch_size=32, num_workers=2, img_size=224, seed=42):
    train_tf, val_tf = get_transforms(img_size)
    dataset, train_idx, val_idx, test_idx = create_splits(data_dir, seed)

    train_ds = Subset(datasets.ImageFolder(root=data_dir, transform=train_tf), train_idx)
    val_ds = Subset(datasets.ImageFolder(root=data_dir, transform=val_tf), val_idx)
    test_ds = Subset(datasets.ImageFolder(root=data_dir, transform=val_tf), test_idx)

    # Weighted sampler to mitigate class imbalance
    train_targets = [dataset.targets[i] for i in train_idx]
    num_classes = len(dataset.classes)
    class_counts = [train_targets.count(i) for i in range(num_classes)]
    class_weights = 1.0 / (np.array(class_counts) + 1e-9)
    sample_weights = [class_weights[t] for t in train_targets]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, dataset.classes