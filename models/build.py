from .conv2next import conv2next_tiny
from torchvision import models
from timm.models.vision_transformer import VisionTransformer
import torch

def build_model(config):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    num_classes = config.MODEL.NUM_CLASSES

    if model_type == 'resnet18':
        model = models.resnet18(pretrained=False, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=False, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_type == 'conv2next-tiny':
        model = conv2next_tiny(num_classes=num_classes)
    elif model_type == 'vit-tiny':
        model = VisionTransformer(img_size=config.DATA.IMG_SIZE, patch_size=config.MODEL.PATCH_SIZE, num_classes=num_classes,
                                  embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE)
    elif model_type == 'CNN':
        raise NotImplementedError(f"Model not implemented yet: {model_type}")
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    if config.MODEL.NAME.lower() == 'wn':
        print("Applying weight normalization to the model.")
        modules = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        modules = modules[1:-1]
        for m in modules:
            if hasattr(m, 'weight'):
                torch.nn.utils.weight_norm(m, name='weight', dim=0)
    elif config.MODEL.NAME.lower() == 'sn':
        print("Applying spectral normalization to the model.")
        modules = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        modules = modules[1:-1]
        for m in modules:
            if hasattr(m, 'weight'):
                torch.nn.utils.spectral_norm(m, name='weight', n_power_iterations=1)

    return model
