import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from typing import Optional
from torch.nn.parallel import DistributedDataParallel as DDP


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Positional Embedding (RoPE) to input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (..., seq_len, dim)
        rope (torch.Tensor): Precomputed rotation matrices of shape (seq_len, dim)

    Returns:
        torch.Tensor: Rotated tensor
    """
    return torch.einsum('...nd,sd->...ns', x, rope)


def get_rope_matrix(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Generates RoPE rotation matrix using sinusoidal encoding.

    Args:
        seq_len (int): Sequence length
        dim (int): Feature dimension (should be even)
        device (torch.device): Device to place the matrix on

    Returns:
        torch.Tensor: RoPE matrix of shape (seq_len, dim)
    """
    assert dim % 2 == 0, "Dimension must be even for RoPE."
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    sinusoid = torch.einsum("i,j->ij", positions, inv_freq)
    sin, cos = sinusoid.sin(), sinusoid.cos()
    rope = torch.cat([cos, sin], dim=-1)
    return rope


class MultiHeadLinearAttention(nn.Module):
    """
    Multi-head Linear Attention module with support for VO-RoPE (RoPE on Value and Output),
    sharding, and multi-GPU parallelism via DistributedDataParallel (DDP).

    Args:
        dim (int): Input and output dimension
        heads (int): Number of attention heads
        dropout (float): Dropout probability
        device (Optional[torch.device]): Device for computation
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1, device: Optional[torch.device] = None):
        super().__init__()
        assert dim % heads == 0, "Dimension must be divisible by number of heads."

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_vo_rope: bool = True,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H = self.heads

        logger.debug(f"Input shape: {x.shape}")

        # Linear projections
        q = self.to_q(x).view(B, T, H, -1).transpose(1, 2)  # (B, H, T, D/H)
        k = self.to_k(x).view(B, T, H, -1).transpose(1, 2)
        v = self.to_v(x).view(B, T, H, -1).transpose(1, 2)

        logger.debug("Computed Q, K, V projections")

        # RoPE
        if use_vo_rope:
            rope = get_rope_matrix(T, self.head_dim, self.device)
            v = apply_rope(v, rope)  # Apply RoPE to V
            logger.debug("Applied RoPE to V")

        # Attention weights (linear variant, not dot-product)
        scores = torch.softmax(q @ k.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        logger.debug("Computed attention scores")

        out = scores @ v

        if use_vo_rope:
            inv_rope = get_rope_matrix(T, self.head_dim, self.device).transpose(0, 1)
            out = apply_rope(out, inv_rope)
            logger.debug("Applied inverse RoPE to output")

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.to_out(self.dropout(out))

    def shard(self):
        """
        Wraps the module with DistributedDataParallel if multiple GPUs are available.
        """
        if torch.cuda.device_count() > 1:
            logger.info(f"Using DistributedDataParallel on {torch.cuda.device_count()} GPUs")
            return DDP(self)
        else:
            logger.info("Running on single device")
            return self


if __name__ == "__main__":
    logger.info("Testing VO-RoPE MultiHeadLinearAttention with sharding support")
    model = MultiHeadLinearAttention(dim=64, heads=8)
    model = model.shard()
    x = torch.randn(2, 128, 64).to(next(model.parameters()).device)
    y = model(x)
    logger.success(f"Output shape: {y.shape}")
