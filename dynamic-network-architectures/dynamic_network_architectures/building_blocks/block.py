from typing import Union, Type, List, Tuple
import torch
from torch.nn import init
from torch.nn.parameter import Parameter
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
import numpy as np
from torch.nn import functional as F
from einops import rearrange
import typing as t
from timm.models.layers import DropPath
import math
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.regularization import DropPath, SqueezeExcite
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from torch.nn.init import xavier_uniform_, constant_

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )

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


class ParNetAtt(nn.Module):
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
        super(ParNetAtt, self).__init__()
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
        self.att = ParNetAttention(conv_op=conv_op, channel=output_channels)

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


# classes
class ParNetAttention(nn.Module):

    def __init__(self, conv_op, channel=512):
        super().__init__()
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.sse = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel, channel, kernel_size=1),
                nn.Sigmoid()
            )

            self.conv1x1 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1),
                nn.BatchNorm2d(channel)
            )
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(channel)
            )
        else:
            self.sse = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(channel, channel, kernel_size=1),
                nn.Sigmoid()
            )

            self.conv1x1 = nn.Sequential(
                nn.Conv3d(channel, channel, kernel_size=1),
                nn.BatchNorm3d(channel)
            )
            self.conv3x3 = nn.Sequential(
                nn.Conv3d(channel, channel, kernel_size=3, padding=1),
                nn.BatchNorm3d(channel)
            )
        self.silu = nn.SiLU()

    def forward(self, x):
        # b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y


class CA(nn.Module):
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
        super(CA, self).__init__()
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
        self.ca = ChannelAttention(conv_op=conv_op, channels=output_channels)

    def forward(self, x):
        x = self.all_modules(x)
        x = self.ca(x)
        # x = self.sa(x)
        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class CBAM(nn.Module):
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
        super(CBAM, self).__init__()
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
        self.ca = ChannelAttention(conv_op=conv_op, channels=output_channels)
        self.sa = SpatialAttention(conv_op=conv_op, kernel_size=5)

    def forward(self, x):
        x = self.all_modules(x)
        ca = self.ca(x)
        sa = self.sa(x)

        return ca + sa

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, conv_op, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        elif conv_op == torch.nn.modules.conv.Conv3d:
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Conv3d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, conv_op, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel size must be 3 or 7"
        if kernel_size in [5, 7]:
            padding = kernel_size // 2
        else:
            padding = 1
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        elif conv_op == torch.nn.modules.conv.Conv3d:
            self.cv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))



class DualConv(nn.Module):
    def __init__(self, conv_op, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, g=4):
        super(DualConv, self).__init__()
        self.pwc = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        # print('using dual conv...', in_channels, g)
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                groups=g, bias=bias, dilation=dilation)
            if kernel_size[0] == 1:
                self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        elif conv_op == torch.nn.modules.conv.Conv3d:
            self.gc = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                groups=g, bias=bias, dilation=dilation)
            if kernel_size[0] == 1:
                self.pwc = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, input_data):
        # if self.pwc:
        #     return self.gc(input_data) + self.pwc(input_data)
        # return self.gc(input_data)
        return self.gc(input_data) + self.pwc(input_data)


class PSA(nn.Module):
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
        super(PSA, self).__init__()
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
        self.ca = ChannelAttention(conv_op=conv_op, channels=output_channels)

    def forward(self, x):
        x = self.all_modules(x)
        x = self.ca(x) * x
        # x = self.sa(x) * x
        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, conv_op, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        elif conv_op == torch.nn.modules.conv.Conv3d:
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Conv3d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))
#
#
# class PolarizedSelfAttention(nn.Module):
#     def __init__(self, conv_op, channels: int):
#         super().__init__()
#         self.is_3d = True
#         if conv_op == torch.nn.modules.conv.Conv2d:
#             self.is_3d = False
#             self.ch_wv = nn.Conv2d(channels, channels // 2, kernel_size=(1, 1))
#             self.ch_wq = nn.Conv2d(channels, 1, kernel_size=(1, 1))
#             # self.softmax_channel=nn.Softmax(1)
#             # self.softmax_spatial=nn.Softmax(-1)
#             self.ch_wz = nn.Conv2d(channels // 2, channels, kernel_size=(1, 1))
#             # self.ln=nn.LayerNorm(channels)
#             # self.sigmoid=nn.Sigmoid()
#             self.sp_wv = nn.Conv2d(channels, channels // 2, kernel_size=(1, 1))
#             self.sp_wq = nn.Conv2d(channels, channels // 2, kernel_size=(1, 1))
#             self.agp = nn.AdaptiveAvgPool2d((1, 1))
#         elif conv_op == torch.nn.modules.conv.Conv3d:
#             self.ch_wv = nn.Conv3d(channels, channels // 2, kernel_size=(1, 1, 1))
#             self.ch_wq = nn.Conv3d(channels, 1, kernel_size=(1, 1, 1))
#             self.ch_wz = nn.Conv3d(channels // 2, channels, kernel_size=(1, 1, 1))
#             self.sp_wv = nn.Conv3d(channels, channels // 2, kernel_size=(1, 1, 1))
#             self.sp_wq = nn.Conv3d(channels, channels // 2, kernel_size=(1, 1, 1))
#             self.agp = nn.AdaptiveAvgPool3d((1, 1, 1))
#         self.softmax_channel = nn.Softmax(1)
#         self.softmax_spatial = nn.Softmax(-1)
#         self.ln = nn.LayerNorm(channels)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         if not self.is_3d:
#             b, c, h, w = x.size()
#
#             # Channel-only Self-Attention
#             channel_wv = self.ch_wv(x)  # bs,c//2,h,w
#             channel_wq = self.ch_wq(x)  # bs,1,h,w
#             channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
#             channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
#             channel_wq = self.softmax_channel(channel_wq)
#             channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
#             channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0,
#                                                                                                                      2,
#                                                                                                                      1).reshape(
#                 b, c, 1, 1)  # bs,c,1,1
#             channel_out = channel_weight * x
#
#             # Spatial-only Self-Attention
#             spatial_wv = self.sp_wv(x)  # bs,c//2,h,w
#             spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
#             spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
#             spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
#             spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
#             spatial_wq = self.softmax_spatial(spatial_wq)
#             spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
#             spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
#             spatial_out = spatial_weight * x
#             out = spatial_out + channel_out
#         else:
#             b, c, h, w, d = x.size()
#
#             # Channel-only Self-Attention
#             channel_wv = self.ch_wv(x)  # bs,c//2,h,w
#             channel_wq = self.ch_wq(x)  # bs,1,h,w
#             channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
#             channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
#             channel_wq = self.softmax_channel(channel_wq)
#             channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1).unsqueeze(-1)  # bs,c//2,1,1, 1
#             channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0,
#                                                                                                                      2,
#                                                                                                                      1).reshape(
#                 b, c, 1, 1, 1)  # bs,c,1,1, 1
#             channel_out = channel_weight * x
#
#             # Spatial-only Self-Attention
#             spatial_wv = self.sp_wv(x)  # bs,c//2,h,w
#             spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
#             spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1, 1
#             spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w*d
#             spatial_wq = spatial_wq.permute(0, 2, 3, 4, 1).reshape(b, 1, c // 2)  # bs,1,c//2
#             spatial_wq = self.softmax_spatial(spatial_wq)
#             spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w*d
#             spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w, d))  # bs,1,h,w
#             spatial_out = spatial_weight * x
#             out = spatial_out + channel_out
#         return out

class SCSA(nn.Module):
    def __init__(
        self,
        dim: int,
        head_num: int = 8,
        window_size: int = 7,
        group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
        qkv_bias: bool = False,
        fuse_bn: bool = False,
        down_sample_mode: str = "avg_pool",
        attn_drop_ratio: float = 0.0,
        gate_layer: str = "sigmoid",
    ):
        super(SCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim % 4 == 0, "The dimension of input feature should be divisible by 4."
        self.group_chans = group_chans = self.dim // 4

        # Define local and global depthwise 3D convolutions
        self.local_dwc = nn.Conv3d(
            group_chans, group_chans, kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2, groups=group_chans
        )
        self.global_dwc_s = nn.Conv3d(
            group_chans, group_chans, kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2, groups=group_chans
        )
        self.global_dwc_m = nn.Conv3d(
            group_chans, group_chans, kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2, groups=group_chans
        )
        self.global_dwc_l = nn.Conv3d(
            group_chans, group_chans, kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2, groups=group_chans
        )

        # Attention gating layers
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == "softmax" else nn.Sigmoid()
        self.norm_d = nn.GroupNorm(4, dim)  # Depth normalization
        self.norm_h = nn.GroupNorm(4, dim)  # Height normalization
        self.norm_w = nn.GroupNorm(4, dim)  # Width normalization

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)  # Channel normalization

        # Define query, key, and value convolutions in 3D
        self.q = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == "softmax" else nn.Sigmoid()

        # Downsampling function
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            if down_sample_mode == "recombination":
                self.down_func = self.space_to_chans
                self.conv_d = nn.Conv3d(dim * window_size**3, dim, kernel_size=1, bias=False)
            elif down_sample_mode == "avg_pool":
                self.down_func = nn.AvgPool3d(kernel_size=(window_size, window_size, window_size), stride=window_size)
            elif down_sample_mode == "max_pool":
                self.down_func = nn.MaxPool3d(kernel_size=(window_size, window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input tensor x with dimensions (B, C, D, H, W)
        """
        b, c, d, h, w = x.size()

        # Compute spatial attention priorities
        x_d = x.mean(dim=4).mean(dim=3)
        l_x_d, g_x_d_s, g_x_d_m, g_x_d_l = torch.split(x_d, self.group_chans, dim=1)

        x_h = x.mean(dim=4).mean(dim=2)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)

        x_w = x.mean(dim=3).mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        # Depth, Height, Width Attention
        x_d_attn = self.sa_gate(self.norm_d(torch.cat((self.local_dwc(l_x_d), self.global_dwc_s(g_x_d_s),
                                                        self.global_dwc_m(g_x_d_m), self.global_dwc_l(g_x_d_l)), dim=1)))
        x_d_attn = x_d_attn.view(b, c, d, 1, 1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((self.local_dwc(l_x_h), self.global_dwc_s(g_x_h_s),
                                                        self.global_dwc_m(g_x_h_m), self.global_dwc_l(g_x_h_l)), dim=1)))
        x_h_attn = x_h_attn.view(b, c, 1, h, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((self.local_dwc(l_x_w), self.global_dwc_s(g_x_w_s),
                                                        self.global_dwc_m(g_x_w_m), self.global_dwc_l(g_x_w_l)), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, 1, w)

        # Apply attention weights
        x = x * x_d_attn * x_h_attn * x_w_attn

        # Channel Attention via Self-Attention
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, d, h, w = y.size()

        # Compute query, key, and value
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        q = rearrange(q, "b (head_num head_dim) d h w -> b head_num head_dim (d h w)", head_num=self.head_num, head_dim=self.head_dim)
        k = rearrange(k, "b (head_num head_dim) d h w -> b head_num head_dim (d h w)", head_num=self.head_num, head_dim=self.head_dim)
        v = rearrange(v, "b (head_num head_dim) d h w -> b head_num head_dim (d h w)", head_num=self.head_num, head_dim=self.head_dim)

        # Self-attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v

        attn = rearrange(attn, "b head_num head_dim (d h w) -> b (head_num head_dim) d h w", d=d, h=h, w=w)
        attn = attn.mean((2, 3, 4), keepdim=True)
        attn = self.ca_gate(attn)

        return attn * x


# =================================ConvNext============================================

# =================================residual============================================
class ConvNext(nn.Module):
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
                 nonlin_first: bool = False,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 ):
        super().__init__()
        self.is_3d = True
        if conv_op == torch.nn.modules.conv.Conv2d:
            self.is_3d = False
            self.dwconv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3,
                                    groups=input_channels)  # depthwise conv
            self.norm = nn.LayerNorm(output_channels, eps=1e-6)
            self.pwconv1 = nn.Linear(output_channels,
                                     4 * output_channels)  # pointwise/1x1 convs, implemented with linear layers
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(4 * output_channels, output_channels)
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((output_channels)),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        elif conv_op == torch.nn.modules.conv.Conv3d:
            self.dwconv = nn.Conv3d(input_channels, output_channels, kernel_size=7, padding=3,
                                    groups=input_channels)  # depthwise conv
            self.norm = nn.LayerNorm(output_channels, eps=1e-6)
            self.pwconv1 = nn.Linear(output_channels,
                                     4 * output_channels)  # pointwise/1x1 convs, implemented with linear layers
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(4 * output_channels, output_channels)
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((output_channels)),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        if self.is_3d:
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
            x = self.norm(x)

            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, D, C) -> (N, C, H, W, D)
            x = input + self.drop_path(x)
        else:
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)

            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            x = input + self.drop_path(x)
        return x
 # =================================ConvNext============================================


# =================================Residual============================================
class Residual(nn.Module):
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
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 # todo wideresnet?
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv1 = ConvDropoutNormReLU(conv_op, input_channels, output_channels, kernel_size, stride, conv_bias,
                                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.conv2 = ConvDropoutNormReLU(conv_op, output_channels, output_channels, kernel_size, 1, conv_bias, norm_op,
                                         norm_op_kwargs, None, None, None, None)

        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x

        # Stochastic Depth
        self.apply_stochastic_depth = False if stochastic_depth_p == 0.0 else True
        if self.apply_stochastic_depth:
            self.drop_path = DropPath(drop_prob=stochastic_depth_p)

        # Squeeze Excitation
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite(self.output_channels, conv_op,
                                                    rd_ratio=squeeze_excitation_reduction_ratio, rd_divisor=8)

        has_stride = (isinstance(stride, int) and stride != 1) or any([i != 1 for i in stride])
        requires_projection = (input_channels != output_channels)

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, adaptive=False, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(conv_op, input_channels, output_channels, 1, 1, False, norm_op,
                                        norm_op_kwargs, None, None, None, None
                                        )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        out += residual
        return self.nonlin2(out)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        # conv1
        output_size_conv1 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # conv2
        output_size_conv2 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # skip conv (if applicable)
        if (self.input_channels != self.output_channels) or any([i != j for i, j in zip(input_size, size_after_stride)]):
            assert isinstance(self.skip, nn.Sequential)
            output_size_skip = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        else:
            assert not isinstance(self.skip, nn.Sequential)
            output_size_skip = 0
        return output_size_conv1 + output_size_conv2 + output_size_skip
# =================================residual============================================
# =================================SE============================================
class SqueezeAndExcitation3D(nn.Module):
    def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation3D, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv3d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool3d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y

class SqueezeAndExciteFusionAdd3D(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd3D, self).__init__()

        self.se_1 = SqueezeAndExcitation3D(channels_in, activation=activation)
        self.se_2 = SqueezeAndExcitation3D(channels_in, activation=activation)

    def forward(self, se1, se2):
        se1 = self.se_1(se1)
        se2 = self.se_2(se2)
        out = se1 + se2
        return out
# =================================SE============================================
# =================================EE============================================
# 论文：PnPNet: Pull-and-Push Networks for Volumetric Segmentation with Boundary Confusion

class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(SDC, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv3dbn(guidance_channels, in_channels, kernel_size=3, padding=1)
        # self.conv1 = Conv3dGN(guidance_channels, in_channels, kernel_size=3, padding=1)
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # initialize
        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        x_initial = self.kernel_initialize(x_initial)

        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, 0, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 0, 2].detach()
        self.x_kernel_diff[:, :, 0, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 2, 2].detach()
        self.x_kernel_diff[:, :, 2, 0, 2].detach()
        self.x_kernel_diff[:, :, 2, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 2, 2].detach()

        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        guidance_initial = self.kernel_initialize(guidance_initial)

        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, 0, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 2].detach()

    def kernel_initialize(self, kernel):
        kernel[:, :, 0, 0, 0] = -1

        kernel[:, :, 0, 0, 2] = 1
        kernel[:, :, 0, 2, 0] = 1
        kernel[:, :, 2, 0, 0] = 1

        kernel[:, :, 0, 2, 2] = -1
        kernel[:, :, 2, 0, 2] = -1
        kernel[:, :, 2, 2, 0] = -1

        kernel[:, :, 2, 2, 2] = 1

        return kernel

    def forward(self, x, guidance):
        guidance_channels = self.guidance_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size

        guidance = self.conv1(guidance)

        x_diff = F.conv3d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1,
                          groups=in_channels)

        guidance_diff = F.conv3d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias,
                                 stride=self.conv.stride, padding=1, groups=in_channels)
        out = self.conv(x_diff * guidance_diff * guidance_diff)
        return out


class SDM(nn.Module):
    def __init__(self, in_channel=4, guidance_channels=4):
        super(SDM, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(in_channel)

    def forward(self, feature, guidance):
        boundary_enhanced = self.sdc1(feature, guidance)
        boundary = self.relu(self.bn(boundary_enhanced))
        boundary_enhanced = boundary + feature

        return boundary_enhanced


class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class Conv3dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)


class Conv3dGNReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        gelu = nn.GELU()

        gn = nn.GroupNorm(4, out_channels)

        super(Conv3dGNReLU, self).__init__(conv, gn, gelu)


class Conv3dGN(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        gn = nn.GroupNorm(4, out_channels)

        super(Conv3dGN, self).__init__(conv, gn)
# =================================SE============================================
class SE(nn.Module):
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
        super(SE, self).__init__()
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
        self.se = SEAttention(conv_op=conv_op, channel=output_channels)

    def forward(self, x):
        x =  self.all_modules(x)
        x = self.se(x)
        return x

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)



class SEAttention(nn.Module):
    def __init__(self, conv_op, channel=512, reduction=16):
        super().__init__()

        if conv_op == torch.nn.modules.conv.Conv2d:
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif conv_op == torch.nn.modules.conv.Conv3d:
            self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            KANLinear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            KANLinear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        size = x.size()
        b, c = size[0], size[1]
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, *[1 for i in range(len(size) - 2)])
        return x * y.expand_as(x)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
#
# class OMC(nn.Module):
#     def __init__(self, conv_op, channel=512, reduction=16):
#         super().__init__()
#
#         if conv_op == torch.nn.modules.conv.Conv2d:
#             self.pool = nn.AdaptiveAvgPool2d(1)
#         elif conv_op == torch.nn.modules.conv.Conv3d:
#             self.pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         size = x.size()
#         b, c = size[0], size[1]
#         y = self.pool(x).view(b, c)
#         y = self.fc(y).view(b, c, *[1 for i in range(len(size) - 2)])
#         return x * y.expand_as(x)
#
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

class Residual(nn.Module):
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
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 # todo wideresnet?
                 ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        self.conv1 = ConvDropoutNormReLU(conv_op, input_channels, output_channels, kernel_size, stride, conv_bias,
                                         norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.conv2 = ConvDropoutNormReLU(conv_op, output_channels, output_channels, kernel_size, 1, conv_bias, norm_op,
                                         norm_op_kwargs, None, None, None, None)

        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x

        # Stochastic Depth
        self.apply_stochastic_depth = False if stochastic_depth_p == 0.0 else True
        if self.apply_stochastic_depth:
            self.drop_path = DropPath(drop_prob=stochastic_depth_p)

        # Squeeze Excitation
        self.apply_se = squeeze_excitation
        if self.apply_se:
            self.squeeze_excitation = SqueezeExcite(self.output_channels, conv_op,
                                                    rd_ratio=squeeze_excitation_reduction_ratio, rd_divisor=8)

        has_stride = (isinstance(stride, int) and stride != 1) or any([i != 1 for i in stride])
        requires_projection = (input_channels != output_channels)

        if has_stride or requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, adaptive=False, pool_type='avg')(stride, stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormReLU(conv_op, input_channels, output_channels, 1, 1, False, norm_op,
                                        norm_op_kwargs, None, None, None, None
                                        )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        if self.apply_stochastic_depth:
            out = self.drop_path(out)
        if self.apply_se:
            out = self.squeeze_excitation(out)
        out += residual
        return self.nonlin2(out)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        # conv1
        output_size_conv1 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # conv2
        output_size_conv2 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # skip conv (if applicable)
        if (self.input_channels != self.output_channels) or any([i != j for i, j in zip(input_size, size_after_stride)]):
            assert isinstance(self.skip, nn.Sequential)
            output_size_skip = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        else:
            assert not isinstance(self.skip, nn.Sequential)
            output_size_skip = 0
        return output_size_conv1 + output_size_conv2 + output_size_skip

# =================================CA FFN============================================
def build_act_layer(act_type):
    # Build activation layer
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()

class ElementScale(nn.Module):
    # A learnable element-wise scaler.
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1, 1)),  # Adjusted for 3D
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class ChannelAggregationFFN(nn.Module):
    def __init__(self,
                 embed_dims,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        # self.feedforward_channels = int(embed_dims * 4)
        self.feedforward_channels = int(96)

        self.fc1 = nn.Conv3d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv3d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv3d(
            in_channels=self.feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv3d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)
        self.SE = SEAttention(conv_op=torch.nn.modules.conv.Conv3d, channel=96)

    def feat_decompose(self, x):
        # x_d: [B, C, D, H, W] -> [B, 1, D, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.SE(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ElementScale(nn.Module):
    #A learnable element-wise scaler.

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class ChannelAggregationFFN(nn.Module):
    def __init__(self,
                 embed_dims,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        # self.feedforward_channels = int(embed_dims * 4)
        self.feedforward_channels = int(96)

        self.fc1 = nn.Conv3d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv3d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv3d(
            in_channels=self.feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv3d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)
        self.SE = SEAttention(conv_op=torch.nn.modules.conv.Conv3d, channel=96)

    def feat_decompose(self, x):
        # x_d: [B, C, D, H, W] -> [B, 1, D, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.SE(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
# =================
import pywt
import pywt.data
from functools import partial

# 论文地址 https://arxiv.org/pdf/2407.05848
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            # 低通滤波
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            # 高通滤波
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseSeparableConvWithWTConv2d, self).__init__()

        # 深度卷积：使用 WTConv2d 替换 3x3 卷积
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)

        # 逐点卷积：使用 1x1 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# =================