import torch

from typing import Callable, Dict

from lib.utils import register

_DISTANCE_FUNC_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = {}

def get_distance_function(distance_name: str) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if distance_name not in _DISTANCE_FUNC_REGISTRY:
        raise ValueError(f"Distance function '{distance_name}' not found in registry.")
    return _DISTANCE_FUNC_REGISTRY[distance_name]

@register(_DISTANCE_FUNC_REGISTRY, 'bhattacharyya')
def bhattacharyya_distance(mean_1: torch.Tensor, std_1: torch.Tensor, mean_2: torch.Tensor, std_2: torch.Tensor) -> torch.Tensor:
    sigma_mean = (std_1 + std_2) / 2.0
    sigma_inv  = 1.0 / (sigma_mean)
    dis1       = torch.sum(torch.pow(mean_1 - mean_2, 2) * sigma_inv, dim=1) / 8.0
    dis2       = 0.5 * (torch.sum(torch.log(sigma_mean), dim=1) - 0.5 * (torch.sum(torch.log(std_1), dim=1) + torch.sum(torch.log(std_2), dim=1)))
    return dis1 + dis2

@register(_DISTANCE_FUNC_REGISTRY, 'hellinger')
def hellinger_distance(mean_1: torch.Tensor, std_1: torch.Tensor, mean_2: torch.Tensor, std_2: torch.Tensor) -> torch.Tensor:
    return torch.pow(1.0 - torch.exp(-bhattacharyya_distance(mean_1, std_1, mean_2, std_2)), 0.5)

@register(_DISTANCE_FUNC_REGISTRY, 'wasserstein')
def wasserstein_distance(mean_1: torch.Tensor, std_1: torch.Tensor, mean_2: torch.Tensor, std_2: torch.Tensor) -> torch.Tensor:
    dis1 = torch.sum(torch.pow(mean_1 - mean_2, 2), dim=1)
    dis2 = torch.sum(torch.pow(torch.pow(std_1, 0.5) - torch.pow(std_2, 0.5), 2), dim=1)
    return torch.pow(dis1 + dis2, 0.5)

@register(_DISTANCE_FUNC_REGISTRY, 'geodesic')
def geodesic_distance(mean_1: torch.Tensor, std_1: torch.Tensor, mean_2: torch.Tensor, std_2: torch.Tensor) -> torch.Tensor:
    u_dis = torch.pow(mean_1 - mean_2, 2)
    std1 = std_1.sqrt()
    std2 = std_2.sqrt()

    sig_dis = torch.pow(std1 - std2, 2)
    sig_sum = torch.pow(std1 + std2, 2)
    delta = torch.div(u_dis + 2 * sig_dis, u_dis + 2 * sig_sum).sqrt()
    return torch.sum(torch.pow(torch.log((1.0 + delta) / (1.0 - delta)), 2) * 2, dim=1).sqrt()

@register(_DISTANCE_FUNC_REGISTRY, 'forward_kl')
def forward_kl_distance(mean_1: torch.Tensor, std_1: torch.Tensor, mean_2: torch.Tensor, std_2: torch.Tensor) -> torch.Tensor:
    return - 0.5 * torch.sum(torch.log(std_1) - torch.log(std_2) - torch.div(std_1, std_2) - torch.div(torch.pow(mean_1 - mean_2, 2), std_2) + 1, dim=1)

@register(_DISTANCE_FUNC_REGISTRY, 'reverse_kl')
def reverse_kl_distance(mean_2: torch.Tensor, std_2: torch.Tensor, mean_1: torch.Tensor, std_1: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(torch.log(std_1) - torch.log(std_2) - torch.div(std_1, std_2) - torch.div(torch.pow(mean_1 - mean_2, 2), std_2) + 1, dim=1)

@register(_DISTANCE_FUNC_REGISTRY, 'jdistance')
def jdistance(mean_1: torch.Tensor, std_1: torch.Tensor, mean_2: torch.Tensor, std_2: torch.Tensor) -> torch.Tensor:
    return forward_kl_distance(mean_1, std_1, mean_2, std_2) + forward_kl_distance(mean_2, std_2, mean_1, std_1)
