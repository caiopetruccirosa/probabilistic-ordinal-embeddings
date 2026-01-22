import torch
import torch.nn as nn

from torchvision.models import VGG

from lib.model.backbone import get_backbone
from lib.model.head import build_age_estimation_head
from lib.model.poe import ProbabilisticOrdinalEmbeddingLayer

from typing import Optional

class AgeEstimationModel(nn.Module):
    def __init__(
        self, 
        backbone_type: str, 
        backbone_imagenet_pretrained: bool, 
        head_type: str, 
        n_age_classes: int,
        use_poe: bool,
        t_samples: Optional[int]
    ):
        super().__init__()
        self.backbone_type                = backbone_type
        self.backbone_imagenet_pretrained = backbone_imagenet_pretrained
        self.head_type                    = head_type
        self.n_age_classes                = n_age_classes
        self.use_poe                      = use_poe
        self.t_samples                    = t_samples

        self.backbone, self.backbone_out_features = get_backbone(backbone_type, backbone_imagenet_pretrained)
        self.head = build_age_estimation_head(head_type, self.backbone_out_features, n_age_classes)
        
        if self.use_poe:
            assert self.t_samples is not None, "t_samples must be provided when using probabilistic ordinal embeddings!"
            self.poe = ProbabilisticOrdinalEmbeddingLayer(self.backbone_out_features, self.backbone_out_features, self.t_samples)
            self.dropout = nn.Dropout()
            if isinstance(self.backbone, VGG):
                self.poe.embedding_mean = nn.Sequential(*list(self.backbone.classifier)[3:5]) # uses penultimate layer (with ReLU) of VGG backbone as embedding mean
                self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier)[:3]) # removes penultimate layer (with ReLU) from VGG backbone
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.backbone(x)
        
        poe = None
        if self.use_poe:
            poe               = self.poe(x)
            _, _, sampled_poe = poe
            x                 = self.dropout(sampled_poe)

        logits, ages = self.head(x)

        return logits, ages, poe
