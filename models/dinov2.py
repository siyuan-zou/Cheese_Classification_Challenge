import torch
import torch.nn as nn


class DinoV2Finetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
                for param in self.backbone.blocks[-1].parameters():
                    param.requires_grad = True
        self.classifier = nn.Linear(self.backbone.norm.normalized_shape[0], num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class DINOv2_Add_Production(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load the pretrained DINOv2 model
        self.backbone = torch.load("checkpoints/20_DINOV2_simple_prompts.pt")
        # self.backbone.head = nn.Identity()  # Replace the classification head with an identity function
        # self.classifier = nn.Linear(self.backbone.norm.normalized_shape[0], num_classes)
        
    def forward(self, x):
        # Forward pass through the model
        x = self.backbone(x)
        # x = self.classifier(x)
        return x