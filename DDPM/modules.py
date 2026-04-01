import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

log = logging.getLogger(__name__)


class EfficientMaskedConv1d(nn.Module):
    """
    1D Convolutional layer with masking.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask=None,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        if mask is None:
            self.layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                padding="same",
                padding_mode=padding_mode,
            )
        else:
            self.layer = MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                mask,
                bias=bias,
                padding_mode=padding_mode,
            )

    def forward(self, x):
        return self.layer.forward(x)


class MaskedConv1d(nn.Module):
    """
    1D Convolutional layer with masking.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        assert (out_channels, in_channels) == mask.shape

        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        total_padding = kernel_size - 1
        left_pad = total_padding // 2
        self.pad = [left_pad, total_padding - left_pad]
        #计算填充大小，并分为左右两部分
        init_k = np.sqrt(1.0 / (in_channels * kernel_size))
        #初始化卷积核权重，给定一个范围
        self.weight = nn.Parameter(
            data=torch.FloatTensor(out_channels, in_channels, kernel_size).uniform_(
                -init_k, init_k
            ),
            requires_grad=True,
        )
        self.register_buffer("mask", mask)#不更新
        self.bias = (
            nn.Parameter(
                data=torch.FloatTensor(out_channels).uniform_(-init_k, init_k),
                requires_grad=True,
            )
            if bias
            else None
        )#初始化偏置项

    def forward(self, x):
        return F.conv1d(
            F.pad(x, self.pad, mode=self.padding_mode),#padding
            self.weight * self.mask.unsqueeze(-1),#应用掩码
            self.bias,
        )


class SLConv(nn.Module):
    """
    Structured Long Convolutional layer.
    Adapted from https://github.com/ctlllll/SGConv

    Args:
        kernel_size: Kernel size used to build convolution.
        num_channels: Number of channels.
        num_scales: Number of scales.
            Overall length will be: kernel_size * (2 ** (num_scales - 1))
        decay_min: Minimum decay. Advanced option.
        decay_max: Maximum decay. Advanced option.
        heads: Number of heads.
        padding_mode: Padding mode. Either "zeros" or "circular".
        use_fft_conv: Whether to use FFT convolution.
        interpolate_mode: Interpolation mode. Either "nearest" or "linear". Advanced option.
    """

    def __init__(
        self,
        kernel_size,
        num_channels,
        num_scales,
        decay_min=2.0,
        decay_max=2.0,
        heads=1,#多头卷积的数量
        padding_mode="zeros",
        use_fft_conv=False,
        interpolate_mode="nearest",
    ):
        super().__init__()
        assert decay_min <= decay_max

        self.h = num_channels
        self.num_scales = num_scales
        self.kernel_length = kernel_size * (2 ** (num_scales - 1))

        self.heads = heads

        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.use_fft_conv = use_fft_conv
        self.interpolate_mode = interpolate_mode

        self.D = nn.Parameter(torch.randn(self.heads, self.h)) #跳跃连接参数

        total_padding = self.kernel_length - 1
        left_pad = total_padding // 2
        self.pad = [left_pad, total_padding - left_pad] #初始化填充

        # Init of conv kernels. There are more options here.
        # Full kernel is always normalized by initial kernel norm.
        self.kernel_list = nn.ParameterList()
        for _ in range(self.num_scales):
            kernel = nn.Parameter(torch.randn(self.heads, self.h, kernel_size))
            self.kernel_list.append(kernel)

        # Support multiple scales. Only makes sense in non-sparse setting.
        self.register_buffer(
            "multiplier",
            torch.linspace(decay_min, decay_max, self.h).view(1, -1, 1),#初始化衰减因子
        )
        self.register_buffer("kernel_norm", torch.ones(self.heads, self.h, 1))#归一化因子
        self.register_buffer(
            "kernel_norm_initialized", torch.tensor(0, dtype=torch.bool)
        )

    def forward(self, x):
        signal_length = x.size(-1)

        kernel_list = []
        for i in range(self.num_scales):
            kernel = F.interpolate(
                self.kernel_list[i],
                scale_factor=2 ** (max(0, i - 1)),
                mode=self.interpolate_mode,
            ) * self.multiplier ** (self.num_scales - i - 1)
            kernel_list.append(kernel)
        k = torch.cat(kernel_list, dim=-1)#拼接所有尺度的卷积

        if not self.kernel_norm_initialized:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(
                1, dtype=torch.bool, device=k.device
            )
            log.debug(f"Kernel norm: {self.kernel_norm.mean()}")
            log.debug(f"Kernel size: {k.size()}")

        assert k.size(-1) < signal_length
        if self.use_fft_conv:
            k = F.pad(k, (0, signal_length - k.size(-1)))

        k = k / self.kernel_norm#归一化

        # Convolution
        if self.use_fft_conv:
            if self.padding_mode == "constant":
                factor = 2 #两端用0填充
            elif self.padding_mode == "circular":
                factor = 1#信号循环

            k_f = torch.fft.rfft(k, n=factor * signal_length)  # (C H L) 卷积核的频域表示
            u_f = torch.fft.rfft(x, n=factor * signal_length)  # (B H L) 输入信号的频域表示
            y_f = torch.einsum("bhl,chl->bchl", u_f, k_f) #频域卷积
            slice_start = self.kernel_length // 2
            y = torch.fft.irfft(y_f, n=factor * signal_length) #变回去

            if self.padding_mode == "constant":
                y = y[..., slice_start : slice_start + signal_length]  # (B C H L)
            elif self.padding_mode == "circular":
                y = torch.roll(y, -slice_start, dims=-1)
            y = rearrange(y, "b c h l -> b (h c) l")

        #这里H是通道数，C是卷积的头数
        else:
            # Pytorch implements convolutions as cross-correlations! flip necessary
            y = F.conv1d(
                F.pad(x, self.pad, mode=self.padding_mode),
                rearrange(k.flip(-1), "c h l -> (h c) 1 l"),
                groups=self.h,
            )#普通卷积

        # Compute D term in state space equation - essentially a skip connection
        y = y + rearrange(
            torch.einsum("bhl,ch->bchl", x, self.D),
            "b c h l -> b (h c) l",
        )

        return y


### Sparse masks


def get_in_mask(
    signal_channel,
    hidden_channel,
    cond_channel=0,
):
    """
    Returns the input mask for the specified mode.

    Args:
        signal_channel: Number of signal channels.
        hidden_channel: Number of hidden channels.
        time_channel: Number of diffusion time embedding channels.
        cond_channel: Number of conditioning channels.
    Returns:
        Input mask as torch tensor.
    """
    np_mask = np.concatenate(
        (
            get_restricted(signal_channel, 1, hidden_channel),#对EEGsignal使用受限连接
            get_full(cond_channel, signal_channel * hidden_channel),#对条件输入用全连接
        ),
        axis=1,
    )
    return torch.from_numpy(np.float32(np_mask))


def get_mid_mask(signal_channel, hidden_channel, off_diag, num_heads=1):
    """
    Returns the hidden mask for the specified mode.

    Args:
        signal_channel: Number of signal channels.
        hidden_channel: Number of hidden channels.
        off_diag: Number of off-diagonal interactions.
        num_heads: Number of heads.

    Returns:
        Mid mask as torch tensor.
    """
    np_mask = np.maximum(
        get_restricted(signal_channel, hidden_channel, hidden_channel),
        get_sub_interaction(signal_channel, hidden_channel, off_diag),
    )

    return torch.from_numpy(np.float32(np.repeat(np_mask, num_heads, axis=1)))


def get_out_mask(signal_channel, hidden_channel):
    """
    Returns the output mask for the specified mode.

    Args:
        signal_channel: Number of signal channels.
        hidden_channel: Number of hidden channels.

    Returns:
        Output mask as torch tensor.
    """
    np_mask = get_restricted(signal_channel, hidden_channel, 1)
    return torch.from_numpy(np.float32(np_mask))


def get_full(num_in, num_out):#全1
    """Get full mask containing all ones."""
    return np.ones((num_out, num_in))


def get_restricted(num_signal, num_in, num_out): 
    """Get mask with ones only on the block diagonal.
    生成对角矩阵，可以改写成这样
    def get_restricted(num_signal, num_in, num_out):
        
        Get mask with ones only on the block diagonal.
        生成对角矩阵
        
        # 第一步：使用 np.eye(num_signal) 生成一个 num_signal x num_signal 的单位矩阵
        # 单位矩阵是一个主对角线元素为 1，其余元素为 0 的方阵
        identity_matrix = np.eye(num_signal)
        
        # 第二步：使用 np.repeat(identity_matrix, num_out, axis=0) 沿着行方向（axis=0）重复单位矩阵
        # 每一行都会重复 num_out 次（相当于列数 *= num_out）
        repeated_rows = np.repeat(identity_matrix, num_out, axis=0)
        
        # 第三步：使用 np.repeat(repeated_rows, num_in, axis=1) 沿着列方向（axis=1）重复上一步得到的矩阵
        # 每一列都会重复 num_in 次(相当于行数 *= num_in)
        result = np.repeat(repeated_rows, num_in, axis=1)
        return result

    """
    return np.repeat(np.repeat(np.eye(num_signal), num_out, axis=0), num_in, axis=1)


def get_sub_interaction(num_signal, size_hidden, num_sub_interaction):
    """Get off-diagonal interactions"""
    sub_interaction = np.zeros((size_hidden, size_hidden))#hidden * hidden
    sub_interaction[:num_sub_interaction, :num_sub_interaction] = 1.0 #限定可交互的范围
    return np.tile(sub_interaction, (num_signal, num_signal))
#使用 np.tile 将 sub_interaction 矩阵沿行和列方向重复 num_signal 次，生成一个更大的矩阵。
#最终矩阵的形状为 (num_signal * size_hidden, num_signal * size_hidden)
