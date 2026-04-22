import os
import torch
import torch.optim as optim
import pickle
from model_unet import UNet
from utils import get_dataloaders, CombinedLoss, iou_score, dice_score

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataloaders – adjust data_dir if your CameraRGB/ is not in current folder
# If folders are directly in the current directory, use data_dir="."
# If they are inside a folder named "data", use data_dir="data"
train_loader, test_loader = get_dataloaders(data_dir=".", batch_size=8, target_size=(128,128))

# Model, loss, optimizer
model = UNet(in_channels=3, out_channels=23).to(device)
criterion = CombinedLoss(weight_focal=0.5, weight_dice=0.5)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# Training loop
num_epochs = 15
best_iou = 0.0
train_losses, train_ious, train_dices = [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    
    # Evaluate on training set
    model.eval()
    total_iou, total_dice = 0.0, 0.0
    with torch.no_grad():
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_iou += iou_score(outputs, masks)
            total_dice += dice_score(outputs, masks)
    avg_iou = total_iou / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    
    train_losses.append(avg_loss)
    train_ious.append(avg_iou)
    train_dices.append(avg_dice)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | mIoU: {avg_iou:.4f} | mDice: {avg_dice:.4f}")
    
    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(model.state_dict(), "models/best_model.pth")
        print(f"  -> Best model saved (mIoU: {best_iou:.4f})")
    
    scheduler.step()

# Save training metrics
with open('training_metrics.pkl', 'wb') as f:
    pickle.dump({'train_losses': train_losses, 'train_ious': train_ious, 'train_dices': train_dices}, f)

# Evaluate on test set
model.eval()
test_iou, test_dice = 0.0, 0.0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        test_iou += iou_score(outputs, masks)
        test_dice += dice_score(outputs, masks)
test_iou /= len(test_loader)
test_dice /= len(test_loader)

print(f"\n=== Final Test Results ===")
print(f"Test mIoU: {test_iou:.4f}")
print(f"Test mDice: {test_dice:.4f}")

with open('test_metrics.txt', 'w') as f:
    f.write(f"mIOU: {test_iou:.4f}\n")
    f.write(f"mDICE: {test_dice:.4f}\n")

assert test_iou >= 0.48 and test_dice >= 0.48, "Performance below 0.48 – adjust hyperparameters or train longer"
