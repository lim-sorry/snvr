from .eswin_transformer import ESwinTransformer

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

    if model_type == 'eswin':
        model = ESwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.ESWIN.PATCH_SIZE,
            in_chans=config.MODEL.ESWIN.IN_CHANS,
            num_classes=config.MODEL.ESWIN.NUM_CLASSES,
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