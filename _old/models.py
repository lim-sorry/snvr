import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34
from torchvision.models.vgg import vgg16, vgg19
from torchvision.models.convnext import convnext_tiny
from torchvision.models.swin_transformer import swin_v2_t, swin_t

def get_model(model_name: str, dataset: str, num_classes: int):
    name, depth = model_name.split('_')
    if name == 'resnet':
        if depth == '18':
            model = resnet18(num_classes=num_classes)
        elif depth == '34':
            model = resnet34(num_classes=num_classes)

        if dataset in ['cifar10', 'cifar100']:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(512, num_classes)
    
    if name == 'vgg':
        if depth == '16':
            model = vgg16(num_classes=num_classes)
        elif depth == '19':
            model = vgg19(num_classes=num_classes)
        
        if dataset in ['cifar10', 'cifar100']:
            model.avgpool = nn.Identity()
            model.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, num_classes)
            )

    if name == 'convnext':
        if depth == 'tiny':
            model = convnext_tiny(num_classes=num_classes)
        elif depth == 'tiny-0.5':
            from torchvision.models.convnext import CNBlockConfig, _convnext
            block_setting = [
                CNBlockConfig(48, 96, 3),
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 9),
                CNBlockConfig(384, None, 3),
            ]
            model = _convnext(block_setting, 0.1, None, True, num_classes=num_classes)
            
        if dataset in ['cifar10', 'cifar100']:
            model.features[0][0] = nn.Conv2d(
                in_channels=model.features[0][0].in_channels,
                out_channels=model.features[0][0].out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            modules_to_replace = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    if module.kernel_size == (7, 7):
                        new_conv = nn.Conv2d(
                            in_channels=module.in_channels,
                            out_channels=module.out_channels,
                            kernel_size=5,
                            stride=1,
                            padding=2,
                            bias=(module.bias is not None),
                            groups=module.groups
                        )
                        # 교체할 모듈 정보 저장
                        modules_to_replace.append((name, new_conv))
            
            # 저장된 모듈 정보로 레이어 교체
            for name, new_conv in modules_to_replace:
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                if parent_name:
                    parent_module = model.get_submodule(parent_name)
                    setattr(parent_module, name.split('.')[-1], new_conv)
                else:
                    setattr(model, name, new_conv)
                        

        
    if name == 'swin-v2':
        if depth == 'tiny':
            model = swin_v2_t(num_classes=num_classes)
        elif depth == 'tiny-0.5':
            from torchvision.models.swin_transformer import _swin_transformer, PatchMergingV2, SwinTransformerBlockV2
            model =  _swin_transformer(
                patch_size=[4, 4],
                embed_dim=48,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=[8, 8],
                stochastic_depth_prob=0.2,
                weights=None,
                progress=True,
                block=SwinTransformerBlockV2,
                downsample_layer=PatchMergingV2,
                num_classes=num_classes,
            )

    if name == 'swin':
        if depth == 'tiny':
            model = swin_t(num_classes=num_classes)
    return model
    
if __name__ == '__main__':
    model = get_model('convnext_tiny', 'cifar100', 100)
    print(model)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)