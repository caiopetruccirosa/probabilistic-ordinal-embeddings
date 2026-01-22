import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

def build_age_estimation_head(head_type: str, in_features: int, n_classes: Optional[int] = None) -> nn.Module:
    if head_type == 'regression':
        return RegressionAgeEstimationHead(in_features)
    elif head_type == 'classification':
        if n_classes is None:
            raise ValueError('n_classes must be provided for classification head')
        return ClassificationAgeEstimationHead(in_features, n_classes)
    elif head_type == 'rank':
        assert n_classes is not None, 'n_classes must be provided for rank-based head!'
        return RankBasedAgeEstimationHead(in_features, n_classes)
    else:
        raise AttributeError(f'Unsupported head type: {head_type}')

class RegressionAgeEstimationHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.fc = nn.Linear(self.in_features, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.relu(self.fc(x))
        ages   = logits.squeeze(dim=-1)
        return logits, ages

class ClassificationAgeEstimationHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.fc = nn.Linear(self.in_features, self.n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(x)
        ages   = torch.argmax(logits, dim=-1) # shape: (batch_size,)
        return logits, ages
    
class RankBasedAgeEstimationHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.fc = nn.Linear(self.in_features, 2 * (self.n_classes - 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits         = self.fc(x)

        rank_logits = logits.view(-1, self.n_classes-1, 2) # shape: (batch_size, C-1, 2)
        rank_pred   = torch.argmax(rank_logits, dim=-1)    # shape: (batch_size, C-1)
        ages        = rank_pred.long().sum(dim=-1)         # shape: (batch_size,)

        return logits, ages