import torch.nn as nn

from torchvision.models import (
    VGG,
    vgg11, vgg11_bn,
    vgg13, vgg13_bn,
    vgg16, vgg16_bn,
    vgg19, vgg19_bn
)

def get_backbone(backbone_type: str, pretrained_imagenet: bool) -> tuple[nn.Module, int]:
    if backbone_type == 'vgg11':
        model = vgg11(pretrained=pretrained_imagenet)
    elif backbone_type == 'vgg11_bn':
        model = vgg11_bn(pretrained=pretrained_imagenet)
    elif backbone_type == 'vgg13':
        model = vgg13(pretrained=pretrained_imagenet)
    elif backbone_type == 'vgg13_bn':
        model = vgg13_bn(pretrained=pretrained_imagenet)
    elif backbone_type == 'vgg16':
        model = vgg16(pretrained=pretrained_imagenet)
    elif backbone_type == 'vgg16_bn':
        model = vgg16_bn(pretrained=pretrained_imagenet)
    elif backbone_type == 'vgg19':
        model = vgg19(pretrained=pretrained_imagenet)
    elif backbone_type == 'vgg19_bn':
        model = vgg19_bn(pretrained=pretrained_imagenet)
    else:
        raise AttributeError(f'Unsupported backbone type: {backbone_type}')
    
    if isinstance(model, VGG):
        model.classifier = nn.Sequential(*list(model.classifier)[:5]) # remove last dropout and fully-connected (for classification) layers
        out_features = 4096 # output features of VGG classifier
    else:
        raise AttributeError(f'Backbone type {backbone_type} is not instance of VGG')
        
    return model, out_features