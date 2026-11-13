"""
ResNet50 transfer learning setup (Phase 3).

We take ResNet50 pretrained on ImageNet, replace its 1000-class output
layer with a 4-class head, and provide helpers to freeze/unfreeze parts
of the network for the two training stages:

  Stage 1: freeze the whole backbone, train ONLY the new head.
  Stage 2: unfreeze layer4 (the last residual block) and fine-tune it
           together with the head at a lower learning rate.

Why freeze/unfreeze? The ImageNet features (edges, textures, shapes)
are already useful; training the head first lets it adapt without
destroying the pretrained weights, and only then do we let the most
specialized block adjust to MRI images with a gentle learning rate.
"""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from src.config import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES,
                pretrained: bool = True) -> nn.Module:
    """
    Create a ResNet50 for our 4 classes.

    Args:
        num_classes: number of output classes (glioma/meningioma/
                     notumor/pituitary).
        pretrained:  if True, load ImageNet weights (transfer learning).

    Returns:
        The model. The final fc layer outputs num_classes logits.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    # Replace the ImageNet classifier with our 4-class head.
    in_features = model.fc.in_features          # 2048 for ResNet50
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Stage 1: freeze everything except the classification head.

    Note on BatchNorm: the BN layers keep updating their running
    statistics (means/variances) during training even when their weights
    are frozen, because requires_grad=False only stops gradient updates.
    This is intentional here: it lets the network recalibrate its
    normalization to the MRI domain while the pretrained conv weights
    stay untouched.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_layer4(model: nn.Module) -> None:
    """
    Stage 2: unfreeze layer4 (the deepest residual block) so it can be
    fine-tuned together with the head at a lower learning rate.
    """
    for param in model.layer4.parameters():
        param.requires_grad = True


def trainable_summary(model: nn.Module) -> str:
    """Human-readable count of trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"trainable {trainable:,} / total {total:,} parameters"
