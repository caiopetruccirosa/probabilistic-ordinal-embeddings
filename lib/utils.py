import yaml
import torch
import torch.nn as nn

from typing import Callable, TypeVar, Any, Literal

T = TypeVar('T')

def register(registry: dict[str, T], *keys: str) -> Callable[[T], T]:
    def decorator(callable: T) -> T:
        for key in keys:
            if key in registry.keys():
                raise Exception(f'Key {key} already registered in registry!')
            registry[key] = callable
        return callable
    return decorator

def init_layer(module: nn.Module):
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        if module.weight is not None:
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(0, 0.01)
        module.bias.data.zero_()

def yaml_load(filepath: str) -> Any:
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)
    
def yaml_dump(obj: Any, filepath: str):
    def _sequence_representer(dumper: yaml.SafeDumper, obj: Any):
        is_sequence = all(isinstance(x, (int, float, str)) for x in obj) or None
        return dumper.represent_sequence('tag:yaml.org,2002:seq', obj, flow_style=is_sequence)

    yaml.SafeDumper.add_representer(list, _sequence_representer)
    yaml.SafeDumper.add_representer(tuple, _sequence_representer)

    with open(filepath, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    
def accuracy_metric(prediction: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes the accuracy metric.
    Args:
        prediction (torch.Tensor): Model predictions of shape (batch_size,).
        targets (torch.Tensor): Ground truth labels of shape (batch_size,).
    Returns:
        float: Accuracy value.
    """
    return (prediction == targets).sum().item() / targets.size(dim=0)

def get_available_device() -> Literal['cpu', 'gpu', 'mps']:
    if torch.cuda.is_available():
        return 'gpu'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'