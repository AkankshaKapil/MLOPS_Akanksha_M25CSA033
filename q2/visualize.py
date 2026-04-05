import matplotlib.pyplot as plt
import numpy as np
import torch

def denormalize_cifar10(image_tensor):
    """Denormalize CIFAR-10 image for visualization"""
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    
    if torch.is_tensor(image_tensor):
        image_np = image_tensor.cpu().numpy()
    else:
        image_np = image_tensor
    
    # Denormalize: image = image * std + mean
    for i in range(3):
        image_np[i] = image_np[i] * std[i] + mean[i]
    
    # Clip to [0,1]
    image_np = np.clip(image_np, 0, 1)
    
    # Convert to HWC format
    return np.transpose(image_np, (1, 2, 0))

def plot_comparison(original, adv_scratch, adv_art, title_prefix=""):
    """Plot comparison of original vs adversarial images"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(denormalize_cifar10(original))
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(denormalize_cifar10(adv_scratch))
    axes[1].set_title("FGSM (From Scratch)")
    axes[1].axis('off')
    
    axes[2].imshow(denormalize_cifar10(adv_art))
    axes[2].set_title("FGSM (IBM ART)")
    axes[2].axis('off')
    
    plt.suptitle(title_prefix)
    plt.tight_layout()
    return fig

def plot_perturbation_effect(epsilons, acc_clean, acc_scratch, acc_art):
    """Plot accuracy vs perturbation strength"""
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, acc_clean, 'g-', label='Clean Accuracy (Baseline)', linewidth=2)
    plt.plot(epsilons, acc_scratch, 'r--', label='FGSM (From Scratch)', linewidth=2, marker='o')
    plt.plot(epsilons, acc_art, 'b-.', label='FGSM (IBM ART)', linewidth=2, marker='s')
    plt.xlabel('Perturbation Strength (ε)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Impact of FGSM Attack on Model Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()