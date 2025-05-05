# the network below works for input vectors of only 2 dimensions. COnvert it into a network that can handle inputs of shape (120, 120, 3, 3). Also, this denoising model is supposed to work in an image-to-image translation framework, so the model needs to be conditioned on anothe input of shape (120, 120, 3, 3): class TimeEmbedding(nn.Module):
#     """Embedding for time step t."""
#     def __init__(self, emb_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(1, emb_dim),
#             nn.SiLU(),  # Swish activation
#             nn.Linear(emb_dim, emb_dim),
#             nn.SiLU()
#         )
#
#     def forward(self, t):
#         t = t.float().view(-1, 1)  # Ensure (batch_size, 1) and convert to float
#         return self.mlp(t)
#
# class ResidualBlock(nn.Module):
#     """Residual block with time embedding."""
#     def __init__(self, d, emb_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(d, d * 2)
#         self.fc2 = nn.Linear(d * 2, d)
#         self.emb_layer = nn.Linear(emb_dim, d * 2)  # To match fc1
#
#         self.activation = nn.SiLU()  # Swish activation
#
#     def forward(self, x, t_emb):
#         residual = x  # Store input for residual connection
#         h = self.fc1(x)
#         h += self.emb_layer(t_emb)  # Add time embedding
#         h = self.activation(h)
#         h = self.fc2(h)
#         return residual + h  # Residual connection
#
# class DenoisingNet(nn.Module):
#     """Residual Denoising Network with Time Embedding."""
#     def __init__(self, h, d, complexity=3, emb_dim=16):
#         super().__init__()
#         self.complexity = complexity
#         self.time_embedding = TimeEmbedding(emb_dim)
#
#         # Initial projection
#         self.input_layer = nn.Linear(h, d)
#
#         # Residual blocks
#         self.res_blocks = nn.ModuleList([ResidualBlock(d, emb_dim) for _ in range(complexity)])
#
#         # Output layer
#         self.output_layer = nn.Linear(d, h)
#
#     def forward(self, x, t):
#         t = t.float()  # Ensure t is a float tensor
#         t_emb = self.time_embedding(t)  # Get time embedding
#
#         x = self.input_layer(x)
#
#         for block in self.res_blocks:
#             x = block(x, t_emb)  # Residual connection
#
#         return self.output_layer(x)

import torch
import torch.nn as nn
from typing import Tuple


def transform_to_pytorch_format(tensor: torch.Tensor) -> torch.Tensor:
    """
    Transforms a tensor of shape (B, H, W, D1, D2) to (B, D1*D2, H, W)
    where B is batch size, H/W are spatial dimensions, and D1/D2 are hypercube dimensions

    Args:
        tensor: Tensor of shape (batch_size, height, width, d1, d2)

    Returns:
        Tensor reshaped to (batch_size, d1*d2, height, width)
    """
    b, h, w, d1, d2 = tensor.shape
    # Reshape to combine hypercube dimensions into channels
    tensor = tensor.reshape(b, h, w, d1 * d2)
    # Move channel dimension to second position (after batch)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def transform_from_pytorch_format(tensor: torch.Tensor) -> torch.Tensor:
    """
    Transforms a tensor of shape (B, C, H, W) back to (B, H, W, D1, D2)
    where C = D1*D2

    Args:
        tensor: Tensor of shape (batch_size, channels, height, width)

    Returns:
        Tensor reshaped to (batch_size, height, width, d1, d2)
    """
    b, c, h, w = tensor.shape
    # Move channel dimension to last position
    tensor = tensor.permute(0, 2, 3, 1)
    # Reshape channels back to hypercube dimensions
    d1, d2 = 3, 3  # Assuming original hypercube dimensions were 3×3
    tensor = tensor.reshape(b, h, w, d1, d2)
    return tensor


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
            kernel_size: Tuple[int, int, int] = (3, 3, 3),
            padding: Tuple[int, int, int] = (1, 1, 1)
    ):
        super().__init__()

        self.conv3d = nn.Conv3d(channels, channels * 2, kernel_size, padding=padding)
        self.conv3d_out = nn.Conv3d(channels * 2, channels, kernel_size, padding=padding)

        self.activation = nn.SiLU()

    def forward(
            self,
            x: torch.Tensor,
            vacuum: torch.Tensor,
            t_emb: torch.Tensor
    ) -> torch.Tensor:
        residual = x

        h = self.conv3d(x)
        vacuum_proj = self.conv3d(vacuum)
        h += vacuum_proj

        b, c, h_, w_, d1, d2 = h.shape
        emb = t_emb.view(b, -1, 1, 1, 1, 1).expand(b, c, h_, w_, d1, d2)
        h += emb

        h = self.activation(h)
        h = self.conv3d_out(h)

        return residual + h


class DenoisingNet(nn.Module):
    """Residual Denoising Network with Time Embedding and Vacuum Conditioning."""

    def __init__(
            self,
            height: int = 120,
            width: int = 120,
            num_channels: int = 9,  # 3 × 3 = 9 channels
            complexity: int = 3,
            emb_dim: int = 16
    ):
        super().__init__()
        self.complexity = complexity
        self.time_embedding = TimeEmbedding(emb_dim)

        # Initial convolutional layer
        self.input_layer = nn.Conv3d(num_channels, num_channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels * 2, emb_dim)
            for _ in range(complexity)
        ])

        # Output layer
        self.output_layer = nn.Conv3d(num_channels * 2, num_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(
            self,
            x: torch.Tensor,
            vacuum: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, height, width, d1, d2)
            vacuum: Vacuum conditioning tensor of same shape as x
            t: Time tensor of shape (batch_size,)

        Returns:
            Output tensor of same shape as input
        """
        t = t.float()
        t_emb = self.time_embedding(t)

        x = self.input_layer(x)

        for block in self.res_blocks:
            x = block(x, vacuum, t_emb)

        return self.output_layer(x)