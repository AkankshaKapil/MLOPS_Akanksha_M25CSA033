import matplotlib.pyplot as plt
import pickle
import os

# Load the saved metrics
if not os.path.exists('training_metrics.pkl'):
    raise FileNotFoundError("training_metrics.pkl not found. Run 3_train_unet.py first.")

with open('training_metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

train_losses = metrics['train_losses']
train_ious   = metrics['train_ious']
train_dices  = metrics['train_dices']
epochs = list(range(1, len(train_losses)+1))

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(epochs, train_losses, 'b-')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(epochs, train_ious, 'r-')
plt.title('mIOU')
plt.xlabel('Epoch')
plt.grid(True)

plt.subplot(1,3,3)
plt.plot(epochs, train_dices, 'g-')
plt.title('mDice')
plt.xlabel('Epoch')
plt.grid(True)

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/training_curves.png")
print("Plot saved to plots/training_curves.png")
plt.show()
