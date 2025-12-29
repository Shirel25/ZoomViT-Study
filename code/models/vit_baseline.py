import torch
import torch.nn as nn
import timm

class ViTBaseline(nn.Module):
    """
    Vision Transformer baseline.
    This model serves as a reference point before introducing zoom mechanisms.
    """
    def __init__(self, 
                 num_classes,
                 img_size,
                 patch_size,
                 model_name="vit_tiny_patch16_224"
                 ):
        super().__init__()

        self.model = timm.create_model(
            model_name, 
            pretrained=False,
            num_classes=num_classes,
            img_size=img_size, 
            patch_size=patch_size 
        )
        # Image → Patches → Tokens → [ CLS | patches ] → Transformer → CLS final → Linear layer → Logits (B, 10)

    def forward(self, x):
        return self.model(x) # logits of shape (B, num_classes)



# Test the ViTBaseline model
if __name__ == "__main__":
    model = ViTBaseline(num_classes=10,
                        img_size=32, # image size 32x32 for CIFAR-10
                        patch_size=8) # patch size 8x8, nb patches (32/8)^2=16
    
    dummy_input = torch.randn(2, 3, 32, 32)
    logits = model(dummy_input)
    print("Logits shape:", logits.shape)
# Expected output: Logits shape: torch.Size([2, 10])
