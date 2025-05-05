import torch
import numpy as np
import torch.nn as nn
from typing import Tuple


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod0 = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5)**2
    alphas_cumprod0 = alphas_cumprod0 / alphas_cumprod0[0]
    betas0 = 1 - (alphas_cumprod0[1:] / alphas_cumprod0[:-1])
    return torch.from_numpy(np.clip(betas0, 0, 0.999)).float()


def compute_alphas_betas(T, device):
    betas = cosine_beta_schedule(timesteps=T).to(device)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    alphas_cumprod = torch.cumprod(alphas,0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - sqrt_alphas_cumprod ** 2)
    return betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, alphas_cumprod


def forward_diffusion(x, t, noise, timesteps=100):
    """Forward diffusion process"""
    device = x.device
    betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, alphas_cumprod = compute_alphas_betas(timesteps, device)
    if noise is None:
        noise = torch.randn_like(x).to(device)
    alphas_t = alphas_bar_sqrt[t].view(-1, 1, 1, 1, 1)
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t].view(-1, 1, 1, 1, 1)
    # t = t.view(-1, 1, 1, 1, 1).long()
    return alphas_t * x + alphas_1_m_t * noise


def denoise(model, noisy_input, vacuum, timesteps=100):
    device = noisy_input.device
    model.eval()
    betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, alphas_cumprod = compute_alphas_betas(timesteps, device)

    with torch.no_grad():
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((noisy_input.size(0),), t, device=device, dtype=torch.long)
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]

            # Model estimates noise
            eps_theta = model(noisy_input, vacuum, t_tensor)

            # Compute mean
            mean = (1 / torch.sqrt(alpha_t)) * (noisy_input - beta_t / torch.sqrt(1 - alpha_cumprod_t) * eps_theta)

            if t > 0:
                noise = torch.randn_like(noisy_input)
                sigma_t = torch.sqrt(beta_t)
                noisy_input = mean + sigma_t * noise
            else:
                noisy_input = mean  # Final denoised output

    return noisy_input


class TimeEmbedding(nn.Module):
    """Embedding for time step t."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU()
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().view(-1, 1)
        return self.mlp(t)


class ResidualBlock(nn.Module):
    """Residual block with time embedding and spatial convolutions."""

    def __init__(
            self,
            channels: int,
            emb_dim: int,
            kernel_size: Tuple[int, int, int] = (3, 3, 1),
            padding: Tuple[int, int, int] = (1, 1, 0)
    ):
        super().__init__()

        self.conv3d = nn.Conv3d(channels, channels * 2, kernel_size, padding=padding)
        self.conv3d_out = nn.Conv3d(channels * 2, channels, kernel_size, padding=padding)

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, vacuum: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x

        h = self.conv3d(x)
        # vacuum_proj = self.conv3d(vacuum)
        # h += vacuum_proj

        b, d1, d2, d, c = h.shape
        emb = t_emb.view(b, -1, 1, 1, 1).expand(b, d1, d2, d, c)
        h += emb

        h = self.activation(h)
        h = self.conv3d_out(h)

        return residual + h


class DenoisingNet(nn.Module):
    """Residual Denoising Network with Time Embedding and Vacuum Maps Conditioning."""

    def __init__(self, num_channels: int = 3, complexity: int = 3, emb_dim: int = 12):
        super().__init__()
        self.complexity = complexity
        self.time_embedding = TimeEmbedding(emb_dim)

        # Initial convolutional layer
        self.input_layer = nn.Conv3d(num_channels, num_channels * 2, kernel_size=(3, 3, 1), padding=(1, 1, 0))

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels * 2, emb_dim)
            for _ in range(complexity)
        ])

        # Output layer
        self.output_layer = nn.Conv3d(num_channels * 2, num_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0))

    def forward(self, x: torch.Tensor, vacuum: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        t_emb = self.time_embedding(t)

        x = self.input_layer(x)
        vacuum = self.input_layer(vacuum)

        for block in self.res_blocks:
            x = block(x, vacuum, t_emb)

        return self.output_layer(x)


# class TimeEmbedding(nn.Module):
#     """Embedding for time step t."""
#
#     def __init__(self, emb_dim: int):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(1, emb_dim),
#             nn.SiLU(),
#             nn.Linear(emb_dim, emb_dim),
#             nn.SiLU()
#         )
#
#     def forward(self, t: torch.Tensor) -> torch.Tensor:
#         return self.mlp(t.view(-1, 1))
#
# class VacuumEncoder(nn.Module):
#     """Encodes vacuum map into a conditioning vector."""
#
#     def __init__(self, num_channels: int, emb_dim: int):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv3d(num_channels, emb_dim, kernel_size=3, padding=1),
#             nn.SiLU(),
#             nn.AdaptiveAvgPool3d(1),  # Global pooling to get a single vector
#             nn.Flatten(),
#             nn.Linear(emb_dim, emb_dim)
#         )
#
#     def forward(self, vacuum: torch.Tensor) -> torch.Tensor:
#         return self.encoder(vacuum)
#
# class ResidualBlock(nn.Module):
#     """Residual block with time and vacuum conditioning using FiLM."""
#
#     def __init__(self, channels: int, emb_dim: int):
#         super().__init__()
#
#         self.conv3d = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
#         self.norm = nn.BatchNorm3d(channels)
#         self.activation = nn.SiLU()
#
#         # FiLM-style modulation (scaling and shifting)
#         self.film_mlp = nn.Linear(emb_dim, 2 * channels)  # Outputs (gamma, beta)
#
#     def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
#         residual = x
#         h = self.conv3d(x)
#         h = self.norm(h)
#
#         # Generate FiLM parameters (gamma, beta)
#         gamma, beta = self.film_mlp(cond_emb).chunk(2, dim=1)  # Split into two parts
#         gamma = gamma.view(x.shape[0], -1, 1, 1, 1)  # Reshape for broadcasting
#         beta = beta.view(x.shape[0], -1, 1, 1, 1)
#
#         # Apply FiLM conditioning
#         h = gamma * h + beta
#         h = self.activation(h)
#
#         return residual + h
#
# class DenoisingNet(nn.Module):
#     """Denoising network for DDPM with vacuum conditioning using FiLM."""
#
#     def __init__(self, num_channels: int = 3, complexity: int = 3, emb_dim: int = 12):
#         super().__init__()
#
#         self.time_embedding = TimeEmbedding(emb_dim)
#         self.vacuum_encoder = VacuumEncoder(num_channels, emb_dim)
#
#         self.input_layer = nn.Conv3d(num_channels, num_channels * 2, kernel_size=3, padding=1)
#
#         self.res_blocks = nn.ModuleList([
#             ResidualBlock(num_channels * 2, emb_dim)
#             for _ in range(complexity)
#         ])
#
#         self.output_layer = nn.Conv3d(num_channels * 2, num_channels, kernel_size=3, padding=1)
#
#     def forward(self, x: torch.Tensor, vacuum: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         t_emb = self.time_embedding(t)
#         vacuum_emb = self.vacuum_encoder(vacuum)  # Extract global vacuum conditioning
#         b, d1, d2, d, c = vacuum_emb.shape
#         t_emb = t_emb.view(b, -1, 1, 1, 1).expand(b, d1, d2, d, c)
#         cond_emb = t_emb + vacuum_emb  # Combine time and vacuum embeddings
#
#         x = self.input_layer(x)
#         for block in self.res_blocks:
#             x = block(x, cond_emb)
#
#         return self.output_layer(x)
