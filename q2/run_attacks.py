import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt

from q2.train_resnet18 import train_resnet18, get_cifar10_loaders, ResNet18ForCIFAR10
from q2.fgsm_scratch import evaluate_fgsm_scratch
from q2.fgsm_art import create_art_classifier, evaluate_fgsm_art
from q2.adversarial_detector import run_detection_experiments
from q2.visualize import plot_comparison, plot_perturbation_effect, denormalize_cifar10

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Train ResNet18
    print("\n" + "="*60)
    print("STEP 1: Training ResNet18 on CIFAR-10")
    print("="*60)
    
    # Load data
    trainloader, testloader = get_cifar10_loaders()
    
    # Train model (or load pre-trained)
    try:
        model = ResNet18ForCIFAR10()
        model.load_state_dict(torch.load('best_resnet18_cifar10.pth'))
        model = model.to(device)
        print("Loaded pre-trained model")
    except:
        model = train_resnet18()
    
    # Evaluate clean accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    clean_acc = 100. * correct / total
    print(f"Clean Test Accuracy: {clean_acc:.2f}%")
    
    # Step 2: FGSM Attack Comparison
    print("\n" + "="*60)
    print("STEP 2: FGSM Attack Comparison")
    print("="*60)
    
    epsilons = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    acc_clean = [clean_acc] * len(epsilons)
    acc_scratch = []
    acc_art = []
    
    # Create ART classifier
    classifier = create_art_classifier(model, device)
    
    for eps in epsilons:
        # From scratch
        scratch_acc = evaluate_fgsm_scratch(model, testloader, eps, device)
        acc_scratch.append(scratch_acc)
        
        # With ART
        art_acc = evaluate_fgsm_art(classifier, testloader, eps, device)
        acc_art.append(art_acc)
        
        print(f"ε={eps:.2f}: Scratch={scratch_acc:.2f}%, ART={art_acc:.2f}%")
    
    # Plot perturbation effect
    fig_perturb = plot_perturbation_effect(epsilons, acc_clean, acc_scratch, acc_art)
    plt.savefig('perturbation_effect.png')
    plt.close()
    
    # Step 3: Visual Comparison
    print("\n" + "="*60)
    print("STEP 3: Generating Visual Comparisons")
    print("="*60)
    
    # Get a batch of test images
    test_iter = iter(testloader)
    images, labels = next(test_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Generate adversarial examples for visualization
    from q2.fgsm_scratch import fgsm_attack_scratch
    from q2.fgsm_art import fgsm_attack_art
    
    eps_viz = 0.1
    adv_scratch = fgsm_attack_scratch(model, images, labels, eps_viz, device)
    adv_art_np = fgsm_attack_art(classifier, images.cpu().numpy(), labels.cpu().numpy(), eps_viz)
    adv_art = torch.from_numpy(adv_art_np).to(device)
    
    # Plot sample comparisons
    num_samples = 5
    for i in range(num_samples):
        fig = plot_comparison(
            images[i], adv_scratch[i], adv_art[i],
            title_prefix=f"Sample {i+1} - True Label: {labels[i].item()}"
        )
        plt.savefig(f'comparison_sample_{i+1}.png')
        plt.close()
    
    # Log to WandB
    wandb.init(project="dlops-assignment5", name="adversarial_attacks")
    
    # Log perturbation effect plot
    wandb.log({"perturbation_effect": wandb.Image(fig_perturb)})
    
    # Log sample images
    for i in range(num_samples):
        wandb.log({
            f"sample_{i+1}_original": wandb.Image(denormalize_cifar10(images[i])),
            f"sample_{i+1}_adversarial_scratch": wandb.Image(denormalize_cifar10(adv_scratch[i])),
            f"sample_{i+1}_adversarial_art": wandb.Image(denormalize_cifar10(adv_art[i])),
        })
    
    # Step 4: Adversarial Detection
    print("\n" + "="*60)
    print("STEP 4: Adversarial Detection Models")
    print("="*60)
    
    detection_results = run_detection_experiments(classifier, trainloader, testloader)
    
    for attack, acc in detection_results.items():
        print(f"Detection accuracy for {attack.upper()}: {acc*100:.2f}%")
        wandb.log({f"detection_accuracy_{attack}": acc * 100})
    
    # Summary Table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"FGSM (ε=0.1) Scratch Accuracy: {acc_scratch[epsilons.index(0.1)]:.2f}%")
    print(f"FGSM (ε=0.1) ART Accuracy: {acc_art[epsilons.index(0.1)]:.2f}%")
    print(f"PGD Detector Accuracy: {detection_results['pgd']*100:.2f}%")
    print(f"BIM Detector Accuracy: {detection_results['bim']*100:.2f}%")
    
    wandb.finish()
    
    print("\n✓ Q2 Complete!")

if __name__ == "__main__":
    main()