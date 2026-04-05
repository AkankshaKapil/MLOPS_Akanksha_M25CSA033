import torch
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

def create_art_classifier(model, device, input_shape=(3, 32, 32), nb_classes=10):
    """Wrap PyTorch model as ART classifier"""
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=nb_classes,
        device_type=device.type,
        clip_values=(-2.5, 2.5)  # Approximate range for normalized CIFAR-10
    )
    
    return classifier

def fgsm_attack_art(classifier, images, labels, epsilon):
    """
    FGSM attack using IBM ART.
    
    Args:
        classifier: ART classifier
        images: Input images (numpy array)
        labels: True labels (numpy array)
        epsilon: Perturbation magnitude
    
    Returns:
        adversarial_images: Perturbed images (numpy array)
    """
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    adversarial_images = attack.generate(x=images, y=labels)
    return adversarial_images

def evaluate_fgsm_art(classifier, testloader, epsilon, device):
    """Evaluate model against FGSM attack using ART"""
    classifier.model.eval()
    correct = 0
    total = 0
    
    for images, labels in testloader:
        images_np = images.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Generate adversarial examples using ART
        adv_images_np = fgsm_attack_art(classifier, images_np, labels_np, epsilon)
        adv_images = torch.from_numpy(adv_images_np).to(device)
        
        # Evaluate
        outputs = classifier.model(adv_images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.to(device)).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy