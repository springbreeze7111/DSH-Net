import torch
from torch import nn
from torch.nn.parameter import Parameter

class ShuffleAtt(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(ShuffleAtt, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        self.conv = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding=[(i - 1) // 2 for i in kernel_size],
            dilation=1,
            bias=conv_bias,
        )
        ops.append(self.conv)

        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_op_kwargs)
            ops.append(self.dropout)

        if norm_op is not None:
            self.norm = norm_op(output_channels, **norm_op_kwargs)
            ops.append(self.norm)

        self.all_modules = nn.Sequential(*ops)
        self.att = ShuffleAttention(conv_op=conv_op, channel=output_channels)

    def forward(self, x):
        x = self.all_modules(x)
        x = self.att(x)
        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)

class ShuffleAttention(nn.Module):
    def __init__(self, conv_op, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.is_3d = True
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.is_3d = False
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
            self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
            self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
            self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
            self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        elif conv_op == torch.nn.modules.conv.Conv3d:
            # 自适应平均池化
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
            self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1, 1))
            self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1, 1))
            self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1, 1))
            self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1, 1))

        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, [nn.Conv2d, nn.Conv3d]):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, [nn.BatchNorm2d, nn.BatchNorm3d]):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups, is_3d):
        if is_3d:
            b, c, h, w, d = x.shape
            x = x.reshape(b, groups, -1, h, w, d)
            x = x.permute(0, 2, 1, 3, 4, 5)
            # flatten
            x = x.reshape(b, -1, h, w, d)
        else:
            b, c, h, w = x.shape
            x = x.reshape(b, groups, -1, h, w)
            x = x.permute(0, 2, 1, 3, 4)
            # flatten
            x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        if not self.is_3d:
            b, c, h, w = x.size()
            # group into subfeatures
            x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

            # channel_split
            x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

            # channel attention
            x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
            x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
            x_channel = x_0 * self.sigmoid(x_channel)

            # spatial attention
            x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
            x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
            x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

            # concatenate along channel axis
            out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
            out = out.contiguous().view(b, -1, h, w)

            # channel shuffle
            out = self.channel_shuffle(out, 2, self.is_3d)
        else:
            b, c, h, w, d = x.size()
            # group into subfeatures
            x = x.view(b * self.G, -1, h, w, d)  # bs*G,c//G,h,w

            # channel_split
            x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

            # channel attention
            x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
            x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
            x_channel = x_0 * self.sigmoid(x_channel)

            # spatial attention
            x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
            x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
            x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

            # concatenate along channel axis
            out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
            out = out.contiguous().view(b, -1, h, w, d)

            # channel shuffle
            out = self.channel_shuffle(out, 2, self.is_3d)
        return out

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)