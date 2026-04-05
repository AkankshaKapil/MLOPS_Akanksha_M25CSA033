import torch
import torch.nn as nn
import numpy as np

def fgsm_attack_scratch(model, images, labels, epsilon, device):
    """
    Fast Gradient Sign Method attack implemented from scratch.
    
    Args:
        model: PyTorch model
        images: Input images (normalized)
        labels: True labels
        epsilon: Perturbation magnitude
        device: Device (cuda/cpu)
    
    Returns:
        adversarial_images: Perturbed images
    """
    # Clone images and enable gradients
    perturbed_images = images.clone().detach().to(device)
    perturbed_images.requires_grad = True
    
    # Forward pass
    outputs = model(perturbed_images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
    
    # Get sign of gradient
    grad_sign = perturbed_images.grad.data.sign()
    
    # Create adversarial examples
    perturbed_images = perturbed_images + epsilon * grad_sign
    
    # Clip to valid range (images are normalized)
    perturbed_images = torch.clamp(perturbed_images, -2.5, 2.5)  # Approximate range for normalized CIFAR-10
    
    return perturbed_images.detach()

def evaluate_fgsm_scratch(model, testloader, epsilon, device):
    """Evaluate model against FGSM attack from scratch"""
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = fgsm_attack_scratch(model, images, labels, epsilon, device)
        
        # Evaluate
        outputs = model(adv_images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy