import torch
import torch.nn as nn
import timm

class ViTBaseline(nn.Module):
    """
    Vision Transformer baseline.
    This model serves as a reference point before introducing 
    zoom-based or importance-guided mechanisms (ZoomViT).
    """
    def __init__(self, 
                 num_classes,
                 img_size,
                 patch_size,
                 model_name="vit_tiny_patch16_224"
                 ):
        super().__init__()
        # --------------------------------------------------
        # Create ViT model using timm
        # --------------------------------------------------
        self.model = timm.create_model(
            model_name, 
            pretrained=False,
            num_classes=num_classes,
            img_size=img_size, 
            patch_size=patch_size 
        )
        # Image → Patches → Tokens → [ CLS | patches ] → Transformer → CLS final → Linear layer → Logits (B, 10)
        
        # --------------------------------------------------
        # Save last attention for visualization if needed
        # --------------------------------------------------
        self.last_attention = None

    def forward(self, x, return_attention=False):
        """Forward pass of the ViT baseline"""
        if not return_attention:
            return self.model(x) # logits of shape (B, num_classes)

        # --------------------------------------------------
        # Register hook to capture attention from last block
        # --------------------------------------------------
        self.last_attention = None
        handle = self.model.blocks[-1].attn.register_forward_hook(
            self._save_attention
        )

        # Standard forward pass
        logits = self.model(x)

        # Remove hook after forward
        handle.remove()

        return logits, self.last_attention

    def _save_attention(self, module, input, output):
        """
        Hook function to save attention weights.
        """
        self.last_attention = output

# --------------------------------------------------
# Minimal test (sanity check)
# --------------------------------------------------
if __name__ == "__main__":
    model = ViTBaseline(num_classes=10,
                        img_size=32, # image size 32x32 for CIFAR-10
                        patch_size=8) # patch size 8x8, nb patches (32/8)^2=16
    
    dummy_input = torch.randn(2, 3, 32, 32)
    # Test standard forward
    logits = model(dummy_input)
    print("---- Without attention ----")
    print("Logits shape:", logits.shape)
# Expected output: Logits shape: torch.Size([2, 10])

    # Test forward with attention extraction
    logits, attention = model(dummy_input, return_attention=True)
    print("---- With attention ----")
    print("Logits shape:", logits.shape)
    print("Attention shape:", attention.shape)
# Expected: (batch_size, num_tokens, embed_dim)
        # Logits shape: torch.Size([2, 10])
        # Attention shape: torch.Size([2, 17, 192])
