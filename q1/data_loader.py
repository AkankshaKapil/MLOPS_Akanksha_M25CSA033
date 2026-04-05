import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

def get_cifar100_loaders(batch_size=4, num_workers=2, val_split=0.1, subset_fraction=1.0):
    """
    Load CIFAR-100 dataset with train/val split.
    
    Args:
        batch_size: Number of samples per batch (default 4 for GPU memory)
        num_workers: Number of subprocesses for data loading
        val_split: Fraction of training data to use for validation
        subset_fraction: Use only a fraction of training data (0.0 to 1.0) for quick testing
    """
    train_transform = transforms.Compose([
        transforms.Resize(224),                     
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(224),                     
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load full training dataset
    full_train = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    # Optionally take a subset for quick experimentation
    if subset_fraction < 1.0:
        subset_size = int(len(full_train) * subset_fraction)
        indices = np.random.permutation(len(full_train))[:subset_size]
        full_train = Subset(full_train, indices)
        print(f"Using subset: {subset_size}/{len(full_train)} training samples")
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size]
    )
    
    # Apply validation transform (the dataset inside Subset needs careful handling)
    if hasattr(val_dataset, 'dataset'):
        val_dataset.dataset.transform = val_transform
    else:
        val_dataset.transform = val_transform
    
    # Load test dataset (full, no subset)
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader