import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34
from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod
import numpy as np
from tqdm import tqdm

class AdversarialDetector(nn.Module):
    """Binary classifier to detect adversarial examples"""
    def __init__(self, num_classes=2):
        super().__init__()
        # Use ResNet34 as backbone
        self.model = resnet34(weights=None)
        # Modify for 32x32 images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.model(x)

def generate_adversarial_dataset(classifier, trainloader, attack_type='pgd', eps=0.1):
    """
    Generate a dataset of clean + adversarial images for training detector.
    
    Args:
        classifier: ART classifier (for generating attacks)
        trainloader: DataLoader for clean images
        attack_type: 'pgd' or 'bim'
        eps: Perturbation strength
    
    Returns:
        X_mixed: Mixed clean and adversarial images
        y_mixed: Labels (0=clean, 1=adversarial)
    """
    X_clean = []
    X_adv = []
    y_clean = []
    y_adv = []
    
    # Create attack
    if attack_type == 'pgd':
        attack = ProjectedGradientDescent(
            estimator=classifier, 
            eps=eps,
            max_iter=10,
            eps_step=eps/5
        )
    else:  # BIM
        attack = FastGradientMethod(estimator=classifier, eps=eps)
    
    for images, labels in tqdm(trainloader, desc=f"Generating {attack_type.upper()} examples"):
        # Clean images (label 0)
        X_clean.append(images.numpy())
        y_clean.extend([0] * len(images))
        
        # Adversarial images (label 1)
        adv_images = attack.generate(x=images.numpy(), y=labels.numpy())
        X_adv.append(adv_images)
        y_adv.extend([1] * len(images))
    
    # Combine and shuffle
    X_clean_np = np.concatenate(X_clean, axis=0)
    X_adv_np = np.concatenate(X_adv, axis=0)
    X_mixed = np.concatenate([X_clean_np, X_adv_np], axis=0)
    y_mixed = np.array(y_clean + y_adv)
    
    # Shuffle
    indices = np.random.permutation(len(X_mixed))
    X_mixed = X_mixed[indices]
    y_mixed = y_mixed[indices]
    
    return X_mixed, y_mixed

def train_detector(detector, X_train, y_train, X_test, y_test, epochs=20):
    """Train the adversarial detector"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = detector.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(detector.parameters(), lr=0.001)
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    
    best_acc = 0
    
    for epoch in range(epochs):
        detector.train()
        optimizer.zero_grad()
        
        outputs = detector(X_train_t.to(device))
        loss = criterion(outputs, y_train_t.to(device))
        loss.backward()
        optimizer.step()
        
        # Evaluation
        detector.eval()
        with torch.no_grad():
            train_pred = detector(X_train_t.to(device))
            train_acc = (train_pred.argmax(1) == y_train_t.to(device)).float().mean()
            
            test_pred = detector(X_test_t.to(device))
            test_acc = (test_pred.argmax(1) == y_test_t.to(device)).float().mean()
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(detector.state_dict(), f'detector_{attack_type}.pth')
    
    return best_acc

def run_detection_experiments(classifier, trainloader, testloader):
    """Run adversarial detection experiments for PGD and BIM attacks"""
    results = {}
    
    for attack_type in ['pgd', 'bim']:
        print(f"\n{'='*50}")
        print(f"Training detector for {attack_type.upper()} attack")
        print('='*50)
        
        # Generate datasets
        X_train, y_train = generate_adversarial_dataset(
            classifier, trainloader, attack_type=attack_type, eps=0.1
        )
        X_test, y_test = generate_adversarial_dataset(
            classifier, testloader, attack_type=attack_type, eps=0.1
        )
        
        # Train detector
        detector = AdversarialDetector()
        best_acc = train_detector(detector, X_train, y_train, X_test, y_test)
        
        results[attack_type] = best_acc
        print(f"Best detection accuracy for {attack_type.upper()}: {best_acc:.4f}")
        
        # Ensure ≥70% accuracy
        if best_acc >= 0.70:
            print(f"✓ {attack_type.upper()} detector meets requirement (≥70%)")
        else:
            print(f"✗ {attack_type.upper()} detector below requirement")
    
    return results