import torch
from torch import nn
from typing import Optional, Union, Tuple
from loguru import logger
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_device_and_dtype(
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.device, torch.dtype]:
    """Determine optimal device and dtype based on system capabilities"""
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if dtype is None:
        dtype = (
            torch.float16 if device.type == "cuda" else torch.float32
        )
    return device, dtype


def build_rope_matrix(
    dim: int, seq_len: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    构建 RoPE 的旋转矩阵，形状为 (seq_len, dim)

    Args:
        dim (int): 单个头的特征维度，必须为偶数
        seq_len (int): 序列长度
        device (Optional[torch.device], optional): 放置位置，默认跟随输入张量

    Returns:
        torch.Tensor: (seq_len, dim) 的旋转嵌入
    """
    logger.debug(
        f"Building RoPE matrix: dim={dim}, seq_len={seq_len}"
    )
    if dim % 2 != 0:
        raise ValueError("RoPE embedding dimension must be even.")

    position = torch.arange(
        seq_len, dtype=torch.float, device=device
    ).unsqueeze(1)
    dim_pair = torch.arange(
        0, dim, 2, dtype=torch.float, device=device
    ).unsqueeze(0)
    theta = 10000 ** (-dim_pair / dim)

    freqs = position * theta
    emb = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    return emb


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """
    对输入张量应用 RoPE

    Args:
        x (torch.Tensor): 输入张量 (batch, seq_len, dim)
        rope (torch.Tensor): 预计算好的旋转矩阵 (seq_len, dim)

    Returns:
        torch.Tensor: 应用 RoPE 后的张量
    """
    logger.debug(
        f"Applying RoPE: x.shape={x.shape}, rope.shape={rope.shape}"
    )
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = rope[..., ::2]
    sin = rope[..., 1::2]
    return torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
    )


def apply_inverse_rope(
    x: torch.Tensor, rope: torch.Tensor
) -> torch.Tensor:
    """
    应用 RoPE 的逆旋转

    Args:
        x (torch.Tensor): 输出张量 (batch, seq_len, dim)
        rope (torch.Tensor): 旋转嵌入 (seq_len, dim)

    Returns:
        torch.Tensor: 应用逆旋转后的张量
    """
    logger.debug(
        f"Applying inverse RoPE: x.shape={x.shape}, rope.shape={rope.shape}"
    )
    return apply_rope(x, -rope)


class VORoPEAttention(nn.Module):
    """
    实现 VO-RoPE（第二类旋转位置编码）机制的注意力模块。

    在传统 QK-RoPE 的基础上，进一步在 V 和 O（output）上也应用旋转编码。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        device, dtype = get_device_and_dtype(device, dtype)
        self.device = device
        self.dtype = dtype

        assert self.head_dim % 2 == 0, "每个头的维度必须为偶数"

        self.q_proj = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.k_proj = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.v_proj = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.o_proj = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self._init_weights()
        logger.info(
            f"Initialized VORoPEAttention: dim={dim}, heads={num_heads}, device={device}, dtype={dtype}"
        )

    def _init_weights(self):
        """Initialize weights using standard techniques"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with masking support

        Args:
            x: Input tensor (batch, seq_len, dim)
            attention_mask: Optional attention mask (batch, num_heads, seq_len, seq_len)
            key_padding_mask: Optional padding mask (batch, seq_len)
        """
        B, N, _ = x.shape

        # Move input to correct device/dtype if needed
        x = x.to(device=self.device, dtype=self.dtype)

        rope = build_rope_matrix(self.head_dim, N, device=self.device)
        rope = rope.unsqueeze(0).repeat(B * self.num_heads, 1, 1)

        # Project and reshape
        q = (
            self.q_proj(x)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE
        q = apply_rope(
            q.reshape(B * self.num_heads, N, self.head_dim), rope
        ).view(B, self.num_heads, N, self.head_dim)
        k = apply_rope(
            k.reshape(B * self.num_heads, N, self.head_dim), rope
        ).view(B, self.num_heads, N, self.head_dim)
        v = apply_rope(
            v.reshape(B * self.num_heads, N, self.head_dim), rope
        ).view(B, self.num_heads, N, self.head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Apply masks if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Compute output
        out = torch.matmul(attn_probs, v)

        # Apply inverse RoPE
        out = apply_inverse_rope(
            out.reshape(B * self.num_heads, N, self.head_dim), rope
        ).view(B, self.num_heads, N, self.head_dim)

        # Final projection
        out = out.transpose(1, 2).reshape(B, N, self.dim)
        out = self.o_proj(out)
        out = self.output_dropout(out)

        return out

    @staticmethod
    def setup_distributed():
        """Setup distributed training if not already initialized"""
        if not dist.is_initialized():
            if torch.cuda.is_available():
                dist.init_process_group(backend="nccl")
            else:
                dist.init_process_group(backend="gloo")

    def to_distributed(self) -> DDP:
        """Convert model to distributed version"""
        self.setup_distributed()
        return DDP(self)


if __name__ == "__main__":
    # Example usage with various configurations
    device, dtype = get_device_and_dtype()

    x = torch.randn(1, 10, 1024, device=device, dtype=dtype)
    attention_mask = torch.zeros(
        1, 1, 10, 10, device=device, dtype=dtype
    )
    key_padding_mask = torch.zeros(
        1, 10, device=device, dtype=torch.bool
    )

    attention = VORoPEAttention(
        dim=1024,
        num_heads=16,
        dropout=0.1,
        device=device,
        dtype=dtype,
    )

    # Optional: Convert to distributed
    if torch.cuda.device_count() > 1:
        attention = attention.to_distributed()

    out = attention(x, attention_mask, key_padding_mask)
    print(f"Output shape: {out.shape}")
    print(f"Device: {out.device}, Dtype: {out.dtype}")
