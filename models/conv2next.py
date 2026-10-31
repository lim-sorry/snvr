import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class SpatialAtt(nn.Module):

    def __init__(self, kernel_size=49):
        super().__init__()
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        b, c, _, _ = x.size()
        # residual = x
        x = x * self.sa(x)
        return x

class mixSACA(nn.Module):
    def __init__(self, inp, kernel_size=7,reduction=32,layer_scale_init_value=1e-4):
        super().__init__()
        self.dim = inp // 2
        self.ca = CoordAtt(self.dim, self.dim, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.shuf = Channel_shuffle(self.dim)
    def forward(self, x):
        residual = x
        sp = torch.split(x, self.dim, dim=1)
        sp1 = sp[0]
        sp2 = sp[1]

        sp1 = self.ca(sp1)
        sp2 = sp2*self.sa(sp2)

        output = torch.cat((sp1, sp2),1)

        x = output + residual
        x = self.shuf(x)
        return x


class Channel_shuffle(nn.Module):
    def __init__(self,group):
        super(Channel_shuffle,self).__init__()
        self.groups=group

    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups,
                   channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used befogitre view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x


class Block(nn.Module):
    r""" Conv2NeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, padding=1)  # depthwise conv
        self.bn0 = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, padding=1)  # depthwise conv
        self.bn1 = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # dw1
        x = self.dwconv(x)
        x = self.bn0(x)

        inner = x
        # 1x1 -> 1x1
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)

        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = inner + x

        # dw2
        x = self.dwconv2(x)
        x = self.bn1(x)

        x = input + self.drop_path(x)
        return x

class Conv2NeXt(nn.Module):
    r""" Conv2NeXt

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 10
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [128, 128, 256, 256]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=10,
                 depths=[3, 3, 9, 3], dims=[128, 128, 256, 256], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., **kwargs,
                 ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=1, stride=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            if i == 1:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                    LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
                )
            else:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
                    mixSACA(dims[i + 1], kernel_size=3, reduction=2)
                )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@register_model
def conv2next_tiny(pretrained=False, **kwargs):
    model = Conv2NeXt(depths=[3, 3, 6, 3], dims=[96, 96, 192, 192], **kwargs)
    return model


@register_model
def conv2next_base(pretrained=False, **kwargs):
    model = Conv2NeXt(depths=[3, 3, 9, 3], dims=[128, 128, 256, 256], **kwargs)
    return model


@register_model
def conv2next_large(pretrained=False, **kwargs):
    model = Conv2NeXt(depths=[3, 3, 27, 3], dims=[196, 196, 384, 384], **kwargs)
    return model
