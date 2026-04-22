import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=23):
        super().__init__()
        # U-Net with ResNet34 encoder pretrained on ImageNet
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
            activation=None          # raw logits (no softmax)
        )
    
    def forward(self, x):
        return self.model(x)
