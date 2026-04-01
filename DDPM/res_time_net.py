import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeAwareResidualPredictor(nn.Module):
    def __init__(self, channel_dim=32, signal_length=400, time_embed_dim=128, max_timesteps=1000, num_layers=2):
        super().__init__()
        self.channel_dim = channel_dim
        self.signal_length = signal_length
        self.max_timesteps = max_timesteps
        self.num_layers = num_layers

        # 1. Sinusoidal Time Embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)

        # 2. Project time embedding to (B, channel_dim, signal_length)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, channel_dim * signal_length),
            nn.ReLU()
        )

        # 3. Deep Conv block for residual prediction with normalization
        layers = []
        in_channels = 2 * channel_dim
        for i in range(num_layers):
            layers.append(nn.Conv1d(in_channels, channel_dim, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(4, channel_dim))  # 归一化
            layers.append(nn.ReLU())
            in_channels = channel_dim
        self.conv_net = nn.Sequential(*layers)

    def forward(self, z50, t):
        """
        z50: shape (B, channel_dim, signal_length)
        t: shape (B,) or scalar int
        """
        B, C, L = z50.shape

        # Embed t → (B, time_embed_dim)
        t_embed = self.time_embedding(t)  # (B, D)
        t_proj = self.time_proj(t_embed)  # (B, C*L)
        t_proj = t_proj.view(B, C, L)     # (B, C, L)

        # Concat along channel dimension
        x = torch.cat([z50, t_proj], dim=1)  # (B, 2C, L)

        # Predict residual at timestep t
        res_t_pred = self.conv_net(x)        # (B, C, L)
        return res_t_pred


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        """
        t: shape (B,) or int scalar
        returns: shape (B, dim)
        """
        if isinstance(t, int):
            t = torch.tensor([t], dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(t, torch.Tensor) and len(t.shape) == 0:
            t = t.unsqueeze(0)

        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        freqs = torch.exp(-emb * torch.arange(half_dim, device=t.device))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)
        return emb