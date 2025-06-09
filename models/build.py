from .eswin_transformer import ESwinTransformer
from torchvision import models

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

    img_size = config.DATA.IMG_SIZE
    num_classes = config.MODEL.NUM_CLASSES

    if model_type == 'resnet':
        if config.MODEL.TYPE == 'resnet18':
            model = models.resnet18(pretrained=False, num_classes=num_classes)
            if img_size != 224:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()  # remove maxpool for smaller input sizes
        elif config.MODEL.TYPE == 'resnet34':
            model = models.resnet34(pretrained=False, num_classes=num_classes)
            if img_size != 224:
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()

    elif model_type == 'eswin':
        model = ESwinTransformer(
            img_size=img_size,
            patch_size=config.MODEL.ESWIN.PATCH_SIZE,
            num_classes=num_classes,
            embed_dim=config.MODEL.ESWIN.EMBED_DIM,
            depths=config.MODEL.ESWIN.DEPTHS,
            num_heads=config.MODEL.ESWIN.NUM_HEADS,
            window_size=config.MODEL.ESWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.ESWIN.MLP_RATIO,
            norm_layer=layernorm,
            )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model