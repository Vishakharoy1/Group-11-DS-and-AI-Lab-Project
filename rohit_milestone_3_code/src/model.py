import torch
import torch.nn as nn
import timm

class DeepfakeDetector(nn.Module):
    def __init__(self, model_name='convnextv2_tiny', pretrained=True, num_classes=2):
        super().__init__()
        # Load the backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Classifier head
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def freeze_backbone(self):
        """Freezes the entire backbone, training only the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_last_stage(self):
        """Unfreezes the last stage of ConvNeXt for fine-tuning."""
        # ConvNeXt has stages. The last one is usually stages.3
        self.freeze_backbone() # freeze all first
        
        if hasattr(self.backbone, 'stages'):
            for param in self.backbone.stages[-1].parameters():
                param.requires_grad = True
        
        # Also ensure head is unfrozen
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes the entire model."""
        for param in self.parameters():
            param.requires_grad = True

    def get_optimizer_param_groups(self, lr_head, lr_backbone):
        """Returns parameter groups with separate learning rates."""
        head_params = []
        backbone_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
                
        return [
            {'params': head_params, 'lr': lr_head},
            {'params': backbone_params, 'lr': lr_backbone}
        ]

class CrossAttentionFusion(nn.Module):
    def __init__(self, rgb_dim, freq_dim, hidden_dim):
        super().__init__()
        # Project both to same dimension
        self.rgb_proj = nn.Linear(rgb_dim, hidden_dim)
        self.freq_proj = nn.Linear(freq_dim, hidden_dim)
        
        # Cross Attention: RGB is the query, Frequency is the key/value
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, rgb_feat, freq_feat):
        # inputs are (B, D)
        q = self.rgb_proj(rgb_feat).unsqueeze(1) # (B, 1, H)
        k = self.freq_proj(freq_feat).unsqueeze(1) # (B, 1, H)
        v = k
        
        attn_out, _ = self.attention(q, k, v) # (B, 1, H)
        attn_out = attn_out.squeeze(1) # (B, H)
        
        # Residual connection
        out = self.out_proj(attn_out) + q.squeeze(1)
        return out

class DualStreamDetector(nn.Module):
    def __init__(self, rgb_model_name='convnextv2_tiny', freq_model_name='resnet18', pretrained=True, num_classes=2, rgb_checkpoint=None):
        super().__init__()
        # RGB Branch
        self.rgb_branch = timm.create_model(rgb_model_name, pretrained=pretrained, num_classes=0)
        rgb_dim = self.rgb_branch.num_features
        
        # If the user provides a trained Phase 1 checkpoint, load it into the RGB branch!
        if rgb_checkpoint:
            import os
            if os.path.exists(rgb_checkpoint):
                ckpt = torch.load(rgb_checkpoint, map_location='cpu')
                state_dict = ckpt['model_state_dict']
                # The state dict has 'backbone.xxx' and 'head.xxx'
                # We only want to load the backbone weights into our rgb_branch
                backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
                self.rgb_branch.load_state_dict(backbone_state)
                print(f"✅ Loaded pretrained Phase 1 RGB weights from {rgb_checkpoint}")
            else:
                print(f"⚠️ Warning: Could not find Phase 1 checkpoint at {rgb_checkpoint}")
        
        # Frequency Branch
        self.freq_branch = timm.create_model(freq_model_name, pretrained=pretrained, in_chans=3, num_classes=0)
        freq_dim = self.freq_branch.num_features
        
        # Fusion
        hidden_dim = 512
        self.fusion = CrossAttentionFusion(rgb_dim, freq_dim, hidden_dim)
        
        # Classifier Head
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def extract_fft(self, x):
        """Computes 2D Fast Fourier Transform and returns normalized magnitude spectrum with High-Pass Filter."""
        # 1. Compute 2D FFT
        fft = torch.fft.fft2(x, norm="ortho")
        # 2. Shift low frequencies to the center
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        # 3. Get magnitude and apply log scale to reduce dynamic range
        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log(magnitude + 1e-8)
        
        # 4. Standardize (mean 0, std 1) per image so pretrained network weights behave nicely
        B, C, H, W = magnitude.shape
        mag_flat = magnitude.view(B, C, -1)
        mag_mean = mag_flat.mean(dim=-1, keepdim=True).unsqueeze(-1)
        mag_std = mag_flat.std(dim=-1, keepdim=True).unsqueeze(-1)
        magnitude = (magnitude - mag_mean) / (mag_std + 1e-5)
        
        # 5. HIGH-PASS FILTER MASK: Zero out the center (Low Frequencies)
        # This blinds the ResNet to the actual shape of the face, forcing it to look ONLY at high-frequency AI grids
        center_h, center_w = H // 2, W // 2
        mask_radius = int(H * 0.15) # 15% of the image size
        magnitude[:, :, center_h - mask_radius : center_h + mask_radius, 
                          center_w - mask_radius : center_w + mask_radius] = 0.0
        
        return magnitude

    def forward(self, x):
        # 1. RGB Features
        rgb_feat = self.rgb_branch(x)
        
        # 2. FFT Features
        freq_img = self.extract_fft(x)
        freq_feat = self.freq_branch(freq_img)
        
        # 3. Fusion (Cross-Attention)
        fused_feat = self.fusion(rgb_feat, freq_feat)
        
        # 4. Classification
        logits = self.head(fused_feat)
        return logits
        
    def freeze_backbone(self):
        # Freeze the RGB ConvNeXt branch (we want to preserve Phase 1 weights)
        for param in self.rgb_branch.parameters(): param.requires_grad = False
        
        # We MUST train the Frequency branch because it is brand new to looking at FFT images!
        for param in self.freq_branch.parameters(): param.requires_grad = True
        
        # Train Fusion and Head
        for param in self.fusion.parameters(): param.requires_grad = True
        for param in self.head.parameters(): param.requires_grad = True

    def unfreeze_last_stage(self):
        self.freeze_backbone()
        if hasattr(self.rgb_branch, 'stages'):
            for param in self.rgb_branch.stages[-1].parameters(): param.requires_grad = True
        if hasattr(self.freq_branch, 'layer4'): # Standard ResNet last block
            for param in self.freq_branch.layer4.parameters(): param.requires_grad = True

    def unfreeze_all(self):
        for param in self.parameters(): param.requires_grad = True

    def get_optimizer_param_groups(self, lr_head, lr_backbone):
        head_params = []
        backbone_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            if 'head' in name or 'fusion' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {'params': head_params, 'lr': lr_head},
            {'params': backbone_params, 'lr': lr_backbone}
        ]


def build_model(model_cfg: dict) -> nn.Module:
    arch = model_cfg.get('architecture', 'baseline')
    
    if arch == 'dual_stream':
        model = DualStreamDetector(
            rgb_model_name=model_cfg.get('name', 'convnextv2_tiny'),
            freq_model_name=model_cfg.get('freq_name', 'resnet18'),
            pretrained=model_cfg.get('pretrained', True),
            num_classes=model_cfg.get('num_classes', 2),
            rgb_checkpoint=model_cfg.get('rgb_checkpoint', None)
        )
    else:
        model = DeepfakeDetector(
            model_name=model_cfg.get('name', 'convnextv2_tiny'),
            pretrained=model_cfg.get('pretrained', True),
            num_classes=model_cfg.get('num_classes', 2)
        )
    return model
