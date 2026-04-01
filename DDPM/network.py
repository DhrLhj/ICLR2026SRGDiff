import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from modules import (
    EfficientMaskedConv1d,
    SLConv,
    get_in_mask,
    get_mid_mask,
    get_out_mask,
)


class CatConvBlock(nn.Module):

    def __init__(
        self,
        hidden_channel_full,
        slconv_kernel_size,
        num_scales,
        heads,
        use_fft_conv,
        padding_mode,
        mid_mask,
    ):
        super().__init__()
        self.block = nn.Sequential(
            SLConv(
                num_channels=hidden_channel_full,
                kernel_size=slconv_kernel_size,
                num_scales=num_scales,
                heads=heads,
                padding_mode=padding_mode,
                use_fft_conv=use_fft_conv,
            ),
            nn.BatchNorm1d(heads * hidden_channel_full),
            nn.GELU(),
            EfficientMaskedConv1d(
                in_channels=heads * hidden_channel_full,
                out_channels=hidden_channel_full,
                kernel_size=1,
                mask=mid_mask,
                bias=False,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(hidden_channel_full),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class CatConv(nn.Module):
    """
    Denoising network with structured, long convolutions.

    This implementation uses a fixed number of layers and  was used for all experiments.
    For a more flexible implementation, see SkipLongConv.
    """

    def __init__(
        self,
        signal_length=100,
        signal_channel=1,
        time_dim=10,
        cond_channel=0,
        hidden_channel=20,
        in_kernel_size=17,
        out_kernel_size=17,
        slconv_kernel_size=17,
        num_scales=5,
        heads=1,
        num_blocks=3,
        num_off_diag=20,
        use_fft_conv=False,
        padding_mode="zeros",
        use_pos_emb=False,
    ):
        """
        Args:
            signal_length: Length of the signals used for training.
            signal_channel: Number of signal channels.
            time_dim: Number of diffusion time embedding dimensions.
            cond_channel: Number of conditioning channels.
            hidden_channel: Number of hidden channels per signal channel.
                Total number of hidden channels will be signal_channel * hidden_channel.
            in_kernel_size: Kernel size of the first convolution.
            out_kernel_size: Kernel size of the last convolution.
            slconv_kernel_size: Kernel size used to create the structured long convolutions.
            num_scales: Number of scales used in the structured long convolutions.
            heads: Number of heads used in the structured long convolutions.
            in_mask_mode: Sparsity used for input convolution.
            num_off_diag: Sparsity used for intermediate convolutions.
            out_mask_mode: Sparsity used for output convolution.
            use_fft_conv: Use FFT convolution instead of standard convolution.
            padding_mode: Padding mode. Either "zeros" or "circular".
            activation_type: Activation function used in the network.
            norm_type: Normalization used in the network.
        """

        super().__init__()
        self.signal_length = signal_length  # train signal length
        self.signal_channel = signal_channel
        self.time_dim = time_dim
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = hidden_channel * signal_channel
        cat_time_dim = 2 * time_dim if use_pos_emb else time_dim
        in_channel = signal_channel + cat_time_dim + cond_channel

        in_mask = get_in_mask(
            signal_channel, hidden_channel, cat_time_dim + cond_channel
        )
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, heads)
        )
        out_mask = get_out_mask(signal_channel, hidden_channel)

        self.conv_in = nn.Sequential(
            EfficientMaskedConv1d(
                in_channels=in_channel,
                out_channels=hidden_channel_full,
                kernel_size=in_kernel_size,
                mask=in_mask,
                bias=False,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(hidden_channel_full),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                CatConvBlock(
                    hidden_channel_full=hidden_channel_full,
                    slconv_kernel_size=slconv_kernel_size,
                    num_scales=num_scales,
                    heads=heads,
                    use_fft_conv=use_fft_conv,
                    padding_mode=padding_mode,
                    mid_mask=mid_mask,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = EfficientMaskedConv1d(
            in_channels=hidden_channel_full,
            out_channels=self.signal_channel,
            kernel_size=out_kernel_size,
            mask=out_mask,
            bias=True,  # last layer
            padding_mode=padding_mode,
        )

    def forward(self, sig, t, cond=None):
        #sig->原始信号（在train的过程中sig是加了noise的原始信号）
        #t->time_vector(全是一个数)
        #cond->前面是掩码，后面是原始数据*掩码
        if cond is not None:
            sig = torch.cat([sig, cond], dim=1)
        #133*282*1001-> 噪声和con结合
        if self.use_pos_emb:
            pos_emb = TimestepEmbedder.timestep_embedding(
                torch.arange(self.signal_length, device=sig.device),
                self.time_dim,
            )
            pos_emb = repeat(pos_emb, "l c -> b c l", b=sig.shape[0])
            sig = torch.cat([sig, pos_emb], dim=1)

        time_emb = TimestepEmbedder.timestep_embedding(t, self.time_dim)
        #timestep嵌入,嵌入成16维度，是config中设定的,32*16
        time_emb = repeat(time_emb, "b t -> b t l", l=sig.shape[2])#133*16*1001
        sig = torch.cat([sig, time_emb], dim=1)#133*298*1001 ->相当于是加了16


        #sig.shape = 32*298*1001 -> 298 = 16(time_embedding) + 94(原始数据+noise) + 188（94（conditional_mask）+94（masked_and_conditional_signal）） 
        #mask用于去除异常值，conditional_mask用于筛选哪些通道需要使用
        sig = self.conv_in(sig)#in和out就是带掩码的卷积,掩码也是固定不变的
        for block in self.blocks:
            sig = block(sig)
        sig = self.conv_out(sig)
        return sig


class GeneralEmbedder(nn.Module):
    def __init__(self, cond_channel, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_channel, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, cond):
        cond = rearrange(cond, "b c l -> b l c")
        cond = self.mlp(cond)
        return rearrange(cond, "b l c -> b c l")


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.  sin的time embedding
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )#这个相当于是补齐
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        #最简单的时间嵌入方法
        return t_emb


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class AdaConvBlock(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_scales = num_scales
        self.mid_mask = mid_mask

        self.conv = SLConv(
            self.kernel_size,
            channel,
            num_scales=self.num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.mlp = nn.Sequential(
            EfficientMaskedConv1d(channel, channel, 1, mask=self.mid_mask),
            nn.GELU(),
            EfficientMaskedConv1d(channel, channel, 1, mask=self.mid_mask),
        )

        self.norm1 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        self.norm2 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel // 3, channel * 6, bias=True),
        )

        self.ada_ln[-1].weight.data.zero_()
        self.ada_ln[-1].bias.data.zero_()

    def forward(self, x, t_cond):
        y = x
        y = self.norm1(y)
        temp = self.ada_ln(rearrange(t_cond, "b c l -> b l c"))
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = rearrange(
            temp, "b l c -> b c l"
        ).chunk(6, dim=1)
        y = modulate(y, shift_tm, scale_tm)
        y = self.conv(y)
        y = x + gate_tm * y

        x = y
        y = self.norm2(y)
        y = modulate(y, shift_cm, scale_cm)
        y = x + gate_cm * self.mlp(y)
        return y


class AdaConv(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channel,
        cond_dim=0,
        hidden_channel=8, #隐藏层的扩展因子
        in_kernel_size=1,
        out_kernel_size=1,
        slconv_kernel_size=17,
        num_scales=5,
        num_blocks=3,
        num_off_diag=8,
        use_pos_emb=False,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.signal_channel = signal_channel
        self.signal_length = signal_length
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = signal_channel * hidden_channel

        in_mask = get_in_mask(signal_channel, hidden_channel, 0)#input->hidden
        #shape = hidden_channel_full * num_channel
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, 1)
        )#hidden->hidden
        out_mask = get_out_mask(signal_channel, hidden_channel)#hidden->output
        #这边三个掩码用于控制卷积的稀疏性，in_mask用于输入层，mid_mask用于中间层，out_mask用于输出层
        self.conv_in = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                AdaConvBlock(
                    slconv_kernel_size,
                    hidden_channel_full,
                    num_scales,
                    signal_length=signal_length,
                    mid_mask=mid_mask,
                    padding_mode=padding_mode,
                    use_fft_conv=use_fft_conv,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = EfficientMaskedConv1d(
            hidden_channel_full,
            signal_channel,
            out_kernel_size,
            out_mask,
            padding_mode=padding_mode,
        )

        self.t_emb = TimestepEmbedder(hidden_channel_full // 3)#这里为什么要除以3？
        if self.use_pos_emb:
            self.pos_emb = TimestepEmbedder(hidden_channel_full // 3)
        if cond_dim > 0:
            self.cond_emb = GeneralEmbedder(cond_dim, hidden_channel_full // 3)

    def forward(self, x, t, cond=None):
        x = self.conv_in(x)

        t_emb = self.t_emb(t.to(x.device))
        t_emb = repeat(t_emb, "b c -> b c l", l=self.signal_length)#bc->bcl，具体可以点进去看repeat的注释

        pos_emb = 0
        if self.use_pos_emb:
            pos_emb = self.pos_emb(torch.arange(self.signal_length).to(x.device))
            pos_emb = repeat(pos_emb, "l c -> b c l", b=x.shape[0])
        #个人理解就是先转秩，然后再复制b份
        cond_emb = 0
        if cond is not None:
            cond_emb = self.cond_emb(cond)

        emb = t_emb + pos_emb + cond_emb
        #逐元素相加，并不是cat，所有为什么要除以3呢？
        for block in self.blocks:
            x = block(x, emb)
        #经过多个中间
        x = self.conv_out(x)#out
        return x



class AdaConvBlockNew(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.conv = SLConv(
            kernel_size,
            channel,
            num_scales=num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.mlp = nn.Sequential(
            EfficientMaskedConv1d(channel, channel, 1, mask=mid_mask),
            nn.GELU(),
            EfficientMaskedConv1d(channel, channel, 1, mask=mid_mask),
        )
        self.norm1 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        self.norm2 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        # emb 调制
        self.ada_ln_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel, channel * 2, bias=True),
        )
        # cond 调制
        self.ada_ln_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel, channel * 2, bias=True),
        )
        # 初始化
        self.ada_ln_emb[-1].weight.data.zero_()
        self.ada_ln_emb[-1].bias.data.zero_()
        self.ada_ln_cond[-1].weight.data.zero_()
        self.ada_ln_cond[-1].bias.data.zero_()

    def forward(self, x, emb, cond):
        # emb 调制
        y = self.norm1(x)
        temp_emb = self.ada_ln_emb(rearrange(emb, "b c l -> b l c"))
        shift_emb, scale_emb = temp_emb.chunk(2, dim=-1)
        shift_emb = rearrange(shift_emb, "b l c -> b c l")
        scale_emb = rearrange(scale_emb, "b l c -> b c l")
        y = modulate(y, shift_emb, scale_emb)
        # cond 调制
        temp_cond = self.ada_ln_cond(rearrange(cond, "b c l -> b l c"))
        shift_cond, scale_cond = temp_cond.chunk(2, dim=-1)
        shift_cond = rearrange(shift_cond, "b l c -> b c l")
        scale_cond = rearrange(scale_cond, "b l c -> b c l")
        y = modulate(y, shift_cond, scale_cond)
        # 主体
        y = self.conv(y)
        y = x + y

        x2 = self.norm2(y)
        x2 = self.mlp(x2)
        out = y + x2
        return out

class AdaConvNew(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channel,
        hidden_channel=8,
        in_kernel_size=1,
        out_kernel_size=1,
        slconv_kernel_size=17,
        num_scales=5,
        num_blocks=3,
        num_off_diag=8,
        use_pos_emb=False,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.signal_channel = signal_channel
        self.signal_length = signal_length
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = signal_channel * hidden_channel

        in_mask = get_in_mask(signal_channel, hidden_channel, 0)
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, 1)
        )
        out_mask = get_out_mask(signal_channel, hidden_channel)

        self.conv_in = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.cond_proj = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                AdaConvBlock_FlLM(
                    slconv_kernel_size,
                    hidden_channel_full,
                    num_scales,
                    signal_length=signal_length,
                    mid_mask=mid_mask,
                    padding_mode=padding_mode,
                    use_fft_conv=use_fft_conv,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = EfficientMaskedConv1d(
            hidden_channel_full,
            signal_channel,
            out_kernel_size,
            out_mask,
            padding_mode=padding_mode,
        )

        self.t_emb = TimestepEmbedder(hidden_channel_full)
        if self.use_pos_emb:
            self.pos_emb = TimestepEmbedder(hidden_channel_full)

    def forward(self, x, t, cond=None):
        x = self.conv_in(x)
        t_emb = self.t_emb(t.to(x.device))
        t_emb = repeat(t_emb, "b c -> b c l", l=self.signal_length)
        pos_emb = 0
        if self.use_pos_emb:
            pos_emb = self.pos_emb(torch.arange(self.signal_length).to(x.device))
            pos_emb = repeat(pos_emb, "l c -> b c l", b=x.shape[0])
        emb = t_emb + pos_emb
        if cond is None:
            cond_proj = torch.zeros_like(x)
        else:
            cond_proj = self.cond_proj(cond)
        for block in self.blocks:
            x = block(x, emb, cond_proj)
        x = self.conv_out(x)
        return x


class AdaConvBlock_FlLM(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.conv = SLConv(
            kernel_size,
            channel,
            num_scales=num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.mlp = nn.Sequential(
            EfficientMaskedConv1d(channel, channel, 1, mask=mid_mask),
            nn.GELU(),
            EfficientMaskedConv1d(channel, channel, 1, mask=mid_mask),
        )
        self.norm1 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        self.norm2 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        # emb 调制
        self.ada_ln_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel, channel * 2, bias=True),
        )
        # cond 调制
        self.ada_ln_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel, channel * 2, bias=True),
        )
        # 初始化
        self.ada_ln_emb[-1].weight.data.zero_()
        self.ada_ln_emb[-1].bias.data.zero_()
        self.ada_ln_cond[-1].weight.data.zero_()
        self.ada_ln_cond[-1].bias.data.zero_()

    def forward(self, x, emb, cond):
        # emb 调制
        y = self.norm1(x)
        temp_emb = self.ada_ln_emb(rearrange(emb, "b c l -> b l c"))
        shift_emb, scale_emb = temp_emb.chunk(2, dim=-1)
        shift_emb = rearrange(shift_emb, "b l c -> b c l")
        scale_emb = rearrange(scale_emb, "b l c -> b c l")
        y = modulate(y, shift_emb, scale_emb)
        # cond 调制
        temp_cond = self.ada_ln_cond(rearrange(cond, "b c l -> b l c"))
        shift_cond, scale_cond = temp_cond.chunk(2, dim=-1)
        shift_cond = rearrange(shift_cond, "b l c -> b c l")
        scale_cond = rearrange(scale_cond, "b l c -> b c l")
        y = modulate(y, shift_cond, scale_cond)
        # 主体
        y = self.conv(y)
        y = x + y

        x2 = self.norm2(y)
        x2 = self.mlp(x2)
        out = y + x2
        return out
    
class AdaConv_Res(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channel,
        hidden_channel=8,
        in_kernel_size=1,
        out_kernel_size=1,
        slconv_kernel_size=17,
        num_scales=5,
        num_blocks=3,
        num_off_diag=8,
        use_pos_emb=False,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.signal_channel = signal_channel
        self.signal_length = signal_length
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = signal_channel * hidden_channel

        in_mask = get_in_mask(signal_channel, hidden_channel, 0)
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, 1)
        )
        out_mask = get_out_mask(signal_channel, hidden_channel)

        self.conv_in = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.cond_proj = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.res_proj = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                AdaConvBlock_Res(
                    slconv_kernel_size,
                    hidden_channel_full,
                    num_scales,
                    signal_length=signal_length,
                    mid_mask=mid_mask,
                    padding_mode=padding_mode,
                    use_fft_conv=use_fft_conv,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = EfficientMaskedConv1d(
            hidden_channel_full,
            signal_channel,
            out_kernel_size,
            out_mask,
            padding_mode=padding_mode,
        )

        self.t_emb = TimestepEmbedder(hidden_channel_full)
        if self.use_pos_emb:
            self.pos_emb = TimestepEmbedder(hidden_channel_full)

    def forward(self, x, t, cond=None, res=None):
        x = self.conv_in(x)
        t_emb = self.t_emb(t.to(x.device))
        t_emb = repeat(t_emb, "b c -> b c l", l=self.signal_length)
        pos_emb = 0
        if self.use_pos_emb:
            pos_emb = self.pos_emb(torch.arange(self.signal_length).to(x.device))
            pos_emb = repeat(pos_emb, "l c -> b c l", b=x.shape[0])
        time_emb = t_emb + pos_emb

        if cond is None:
            cond_proj = torch.zeros_like(x)
        else:
            cond_proj = self.cond_proj(cond)
        if res is None:
            res_proj = torch.zeros_like(x)
        else:
            res_proj = self.res_proj(res)

        for block in self.blocks:
            x = block(x, time_emb, cond_proj, res_proj)
        x = self.conv_out(x)
        return x
    
    
class AdaConvBlock_Res(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.conv = SLConv(
            kernel_size,
            channel,
            num_scales=num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.mlp = nn.Sequential(
            EfficientMaskedConv1d(channel, channel, 1, mask=mid_mask),
            nn.GELU(),
            EfficientMaskedConv1d(channel, channel, 1, mask=mid_mask),
        )
        self.norm1 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        self.norm2 = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        # cond+time混合调制
        self.ada_ln_mix = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel, channel * 2, bias=True),
        )
        # res单独调制
        self.ada_ln_res = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel, channel * 2, bias=True),
        )
        # 初始化
        self.ada_ln_mix[-1].weight.data.zero_()
        self.ada_ln_mix[-1].bias.data.zero_()
        self.ada_ln_res[-1].weight.data.zero_()
        self.ada_ln_res[-1].bias.data.zero_()

    def forward(self, x, time_emb, cond_emb, res_emb):
        # Step 1: cond和time混合
        emb_mix = time_emb + cond_emb
        y = self.norm1(x)
        temp_mix = self.ada_ln_mix(rearrange(emb_mix, "b c l -> b l c"))
        shift_mix, scale_mix = temp_mix.chunk(2, dim=-1)
        shift_mix = rearrange(shift_mix, "b l c -> b c l")
        scale_mix = rearrange(scale_mix, "b l c -> b c l")
        y = modulate(y, shift_mix, scale_mix)
        y = self.conv(y)
        y = x + y

        # Step 2: res单独调制
        y2 = self.norm2(y)
        temp_res = self.ada_ln_res(rearrange(res_emb, "b c l -> b l c"))
        shift_res, scale_res = temp_res.chunk(2, dim=-1)
        shift_res = rearrange(shift_res, "b l c -> b c l")
        scale_res = rearrange(scale_res, "b l c -> b c l")
        y2 = modulate(y2, shift_res, scale_res)
        y2 = self.mlp(y2)
        out = y + y2
        return out
    
    
    
class AdaConvBlock_Res_Small(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.conv = SLConv(
            kernel_size,
            channel,
            num_scales=num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.norm = nn.LayerNorm((channel, signal_length), elementwise_affine=False)
        # 条件门控
        self.cond_gate = nn.Sequential(
            nn.Conv1d(channel, channel, 1),
            nn.Sigmoid()
        )
        # 初始化门控偏置为负值，趋于关闭
        nn.init.constant_(self.cond_gate[0].bias, -2.0)

    def forward(self, x, time_emb, cond_emb, res_emb):
        # 主分支：残差连接
        y = self.norm(x)
        y = self.conv(y)
        y = x + y

        # 条件分支：门控微调
        gate = self.cond_gate(cond_emb)
        y = y + gate * cond_emb  # cond_emb shape需与y一致
        # 残差分支：直接加
        y = y + res_emb
        return y

class AdaConv_Res_Small(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channel,
        hidden_channel=8,
        in_kernel_size=1,
        out_kernel_size=1,
        slconv_kernel_size=17,
        num_scales=5,
        num_blocks=3,
        num_off_diag=8,
        use_pos_emb=False,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.signal_channel = signal_channel
        self.signal_length = signal_length
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = signal_channel * hidden_channel

        in_mask = get_in_mask(signal_channel, hidden_channel, 0)
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, 1)
        )
        out_mask = get_out_mask(signal_channel, hidden_channel)

        self.conv_in = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.cond_proj = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.res_proj = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                AdaConvBlock_Res_Small(
                    slconv_kernel_size,
                    hidden_channel_full,
                    num_scales,
                    signal_length=signal_length,
                    mid_mask=mid_mask,
                    padding_mode=padding_mode,
                    use_fft_conv=use_fft_conv,
                )
                for _ in range(num_blocks)
            ]
        )
        self.conv_out = EfficientMaskedConv1d(
            hidden_channel_full,
            signal_channel,
            out_kernel_size,
            out_mask,
            padding_mode=padding_mode,
        )

        self.t_emb = TimestepEmbedder(hidden_channel_full)
        if self.use_pos_emb:
            self.pos_emb = TimestepEmbedder(hidden_channel_full)

    def forward(self, x, t, cond=None, res=None):
        x = self.conv_in(x)
        t_emb = self.t_emb(t.to(x.device))
        t_emb = repeat(t_emb, "b c -> b c l", l=self.signal_length)
        pos_emb = 0
        if self.use_pos_emb:
            pos_emb = self.pos_emb(torch.arange(self.signal_length).to(x.device))
            pos_emb = repeat(pos_emb, "l c -> b c l", b=x.shape[0])
        time_emb = t_emb + pos_emb

        if cond is None:
            cond_proj = torch.zeros_like(x)
        else:
            cond_proj = self.cond_proj(cond)
        if res is None:
            res_proj = torch.zeros_like(x)
        else:
            res_proj = self.res_proj(res)

        for block in self.blocks:
            x = block(x, time_emb, cond_proj, res_proj)
        x = self.conv_out(x)
        return x
    
    


















