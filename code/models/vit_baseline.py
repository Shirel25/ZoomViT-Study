import torch
import torch.nn as nn
import timm

class ViTBaseline(nn.Module):
    """
    Vision Transformer baseline for CIFAR-10 classification.
    This model serves as a reference point before introducing zoom mechanisms.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = timm.create_model(
            "vit_tiny_patch16_224", # nb patches (32/8)^2=16
            pretrained=False,
            num_classes=num_classes,
            img_size=32, # image size 32x32 for CIFAR-10
            patch_size=8 # patch size 8x8
        )
        # Image → Patches → Tokens → [ CLS | patches ] → Transformer → CLS final → Linear layer → Logits (B, 10)

    def forward(self, x):
        return self.model(x) # logits of shape (B, num_classes)



# Test the ViTBaseline model
if __name__ == "__main__":
    model = ViTBaseline()
    dummy_input = torch.randn(2, 3, 32, 32)

    logits = model(dummy_input)

    print("Logits shape:", logits.shape)
# Expected output: Logits shape: torch.Size([2, 10])
