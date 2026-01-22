import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Tuple, Optional, Callable

from lib.distance import get_distance_function
from lib.utils import register

_LOSS_FUNC_REGISTRY: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {}

def get_loss_function(loss_type: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if loss_type not in _LOSS_FUNC_REGISTRY:
        raise AttributeError(f'Unsupported loss type: {loss_type}')
    return _LOSS_FUNC_REGISTRY[loss_type]

@register(_LOSS_FUNC_REGISTRY, 'rank-based')
def rank_based_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes Rank-Based Cross-Entropy loss using ordinal regression (cumulative link model).
    For C classes, creates C-1 binary classifiers. Classifier k predicts whether class > k.
    Args:
        logits: Tensor of shape (batch_size, 2*(C-1)) with 2 logits per ordinal threshold
        targets: Tensor of shape (batch_size,) with class indices in [0, C-1]
    Returns:
        Cross-entropy loss averaged over all ordinal pairs
    Example:
        C=10 classes, target=3: creates 9 binary targets [1,1,1,0,0,0,0,0,0]
    """
    # number of classifiers (C-1, where C is the number of classes)
    n_classifiers = logits.size(dim=1) // 2 

    # create ordinal targets (through vectorized broadcasting): ordinal_targets[i, k] = 1 if targets[i] > k else 0
    ordinal_targets = (targets.unsqueeze(dim=1) > torch.arange(end=n_classifiers, device=logits.device).unsqueeze(dim=0)).long() # shape (batch_size, C-1)
    
    # reshape logits and ordinal targets to shapes (batch_size * (C-1), 2) and (batch_size * (C-1),) respectively for cross-entropy
    return F.cross_entropy(logits.view(-1, 2), ordinal_targets.view(-1))

@register(_LOSS_FUNC_REGISTRY, 'regression')
def regression_mse_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes Mean Squared Error (MSE) loss between the predicted logits and the target values.
    Args:
        logits (torch.Tensor): Predicted values with shape (batch_size, 1).
        targets (torch.Tensor): Ground truth values with shape (batch_size,).
    Returns:
        torch.Tensor: Computed MSE loss.
    """
    return F.mse_loss(logits, targets.unsqueeze(dim=1))

@register(_LOSS_FUNC_REGISTRY, 'classification')
def classification_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """"Computes the Cross-Entropy loss.
    Args:
        logits: torch.Tensor of shape (batch_size, n_classes).
        labels: torch.Tensor of shape (batch_size,).
    Returns:
        torch.Tensor: Computed Cross-Entropy loss.
    """
    return F.cross_entropy(logits, targets)

class ProbabilisticOrdinalLoss(nn.Module):
    def __init__(self, head_type: Literal['classification', 'regression', 'rank'], distance_name: str, alpha: float = 0.0, beta: float = 0.0, delta: float = 0.0):
        super().__init__()
        self.head_type = head_type
        self.alpha, self.beta, self.delta = alpha, beta, delta
        self.distance_f = get_distance_function(distance_name)

    def forward(
        self,
        logits: torch.Tensor,
        embeddings_mean: torch.Tensor,
        embeddings_log_var: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the combined loss: head loss + alpha * ordinal loss + beta * VIB loss.
        Args:
            logits (torch.Tensor): Predicted logits from the model (shape: [t_samples, batch_size, num_classes]).
            embeddings_mean (torch.Tensor): Mean of the probabilistic embeddings (shape: [batch_size, embedding_dim]).
            embeddings_log_var (torch.Tensor): Log variance of the probabilistic embeddings (shape: [batch_size, embedding_dim]).
            targets (torch.Tensor): Ground truth target values (shape: [batch_size,]).
        Returns:
            A tuple containing:
                - head_loss (torch.Tensor): Loss from the main prediction head.
                - vib_loss (torch.Tensor): Variational Information Bottleneck loss.
                - ordinal_loss (torch.Tensor): Ordinal loss based on embedding distances.
                - loss (torch.Tensor): Total combined loss.
        """
        head_loss_fn = get_loss_function(self.head_type)

        flatten_logits  = logits.view(-1, logits.size(dim=-1))  # shape (t_samples * batch_size, num_classes)
        repeated_target = targets.unsqueeze(dim=0).expand(logits.size(dim=0), -1).contiguous().view(-1)  # shape (t_samples * batch_size,)

        head_loss    = head_loss_fn(flatten_logits, repeated_target)
        ordinal_loss = self.alpha * self._ordinal_loss(embeddings_mean, embeddings_log_var, targets)
        vib_loss     = self.beta * self._vib_loss(embeddings_mean, embeddings_log_var)
        loss         = head_loss + vib_loss + ordinal_loss

        return head_loss, vib_loss, ordinal_loss, loss

    def _vib_loss(
        self,
        embeddings_mean: torch.Tensor,
        embeddings_log_var: torch.Tensor,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
    ) -> torch.Tensor:
        """Computes the Variational Information Bottleneck (VIB) loss as the KL divergence between the learned embedding distribution and a standard normal prior.
        Args:
            embeddings_mean (torch.Tensor): Mean of the probabilistic embeddings (shape: [batch_size, embedding_dim]).
            embeddings_log_var (torch.Tensor): Log variance of the probabilistic embeddings (shape: [batch_size, embedding_dim]).
            clamp_min (float): Minimum value to clamp the log variance for numerical stability.
            clamp_max (float): Maximum value to clamp the log variance for numerical stability.
        Returns:
            torch.Tensor: Computed VIB loss."""
        # clamp for numerical stability. shape (batch_size, embedding_dim)
        embeddings_log_var = torch.clamp(embeddings_log_var, min=clamp_min, max=clamp_max)

        # computes KL divergence per sample -> KL(q || p) = 0.5 * sum_d (mean_d^2 + sigma_d^2 - log(sigma_d^2) - 1)
        kl_divergence_loss = 0.5 * torch.sum(embeddings_mean.square() + embeddings_log_var.exp() - embeddings_log_var - 1.0, dim=-1) # shape (batch_size,)

        return kl_divergence_loss.mean()

    def _ordinal_loss(
        self,
        embeddings_mean: torch.Tensor,       # (B, D)
        embeddings_log_var: torch.Tensor,    # (B, D)
        targets: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        """Computes the ordinal loss based on pairwise distances in the probabilistic embedding space.
        Args:
            embeddings_mean (torch.Tensor): Mean of the probabilistic embeddings (shape: [batch_size, embedding_dim]).
            embeddings_log_var (torch.Tensor): Log variance of the probabilistic embeddings (shape: [batch_size, embedding_dim]).
            targets (torch.Tensor): Ground truth target values (shape: [batch_size,]).
        Returns:
            torch.Tensor: Computed ordinal loss.
        """
        batch_size = embeddings_mean.size(0)
        if batch_size < 2:
            return embeddings_mean.new_zeros(())

        # ---------------------------------------------------------
        # 1. computes pairwise targets distances |y_i - y_j| matrix
        # ---------------------------------------------------------
        targets_distances = (targets.unsqueeze(dim=1) - targets.unsqueeze(dim=0)).abs() # shape (batch_size, batch_size)

        # --------------------------------------------------
        # 2. define anchor, second, third samples from batch
        # --------------------------------------------------
        anchor = torch.arange(batch_size, device=targets.device) # shape (batch_size,)
        second = (anchor + 1) % batch_size                       # shape (batch_size,)

        # selects distances |y_i - y_j| where j is the second point from pairwise distances matrix
        pair12_target_distances = targets_distances[anchor, second] # shape (batch_size,)

        # computes | |y_i - y_j| - |y_i - y_second| | -> absolute difference from "distances from anchor to all j to find the third point" and "distances from anchor to second"
        # the difference purpose is to find the third point that is neither the anchor nor the second point and has different distance from the anchor than the second point
        difference = (targets_distances - pair12_target_distances.unsqueeze(dim=1)).abs() # shape (batch_size, batch_size)

        # mask invalid choices (self & equal distances)
        difference.fill_diagonal_(torch.inf)
        difference[difference == 0] = torch.inf

        # selects the third point as the one with minimum difference
        third = difference.argmin(dim=1) # shape (batch_size,)
        pair13_target_distances = targets_distances[anchor, third] # shape (batch_size,)

        # --------------------------------------------------
        # 3. ordinal direction (sign)
        # --------------------------------------------------
        # if |y_i - y_second| < |y_i - y_third| -> sign = -1 (distance to second should be smaller than distance to third)
        # if |y_i - y_second| > |y_i - y_third| -> sign = +1 (distance to second should be larger than distance to third)
        # if |y_i - y_second| == |y_i - y_third| -> sign = 0 (no ordinal information to learn, the targets are the same)
        sign = torch.sign(pair12_target_distances - pair13_target_distances) # shape (batch_size,)

        valid = sign.abs() > 0
        if not valid.any(): # all distances are equal which means that all targets are the same so there is no ordinal information to learn
            return embeddings_mean.new_zeros(())

        # --------------------------------------------------
        # 4. Probabilistic embedding distances
        # --------------------------------------------------
        embeddings_variance = embeddings_log_var.exp()
        distance_pair12 = self.distance_f(
            embeddings_mean[anchor], embeddings_variance[anchor],
            embeddings_mean[second], embeddings_variance[second],
        )
        distance_pair13 = self.distance_f(
            embeddings_mean[anchor], embeddings_variance[anchor],
            embeddings_mean[third], embeddings_variance[third],
        )

        # --------------------------------------------------
        # 5. margin ranking loss
        # --------------------------------------------------
        violation = (distance_pair13 - distance_pair12) * sign + self.delta
        ordinal_loss = F.relu(violation)[valid]

        return ordinal_loss.mean()
