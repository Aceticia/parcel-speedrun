"""Simple BOLD encoder.

Convention: throughout this module, `padding_mask` is a `[B, T]` boolean
tensor where `True` marks padded (invalid) positions, matching PyTorch's
`key_padding_mask`.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import RotaryEmbeddingCat
from timm.models.eva import EvaAttention


class FFW(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


class SimpleBrainEncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, temporal_latent_dim, ffw_dim, num_heads):
        super().__init__()
        total_dim = input_dim + hidden_dim + temporal_latent_dim
        self.norm1 = nn.RMSNorm(total_dim)
        self.ffw = FFW(total_dim, ffw_dim, total_dim - input_dim)
        self.norm2 = nn.RMSNorm(temporal_latent_dim)
        self.temporal = EvaAttention(
            temporal_latent_dim,
            num_heads,
            qk_norm=True,
            norm_layer=nn.RMSNorm,
            num_prefix_tokens=0,
        )

        self.in_dim, self.hidden_dim, self.t_dim = (
            input_dim,
            hidden_dim,
            temporal_latent_dim,
        )

    def forward(self, x, rope, padding_mask: Optional[torch.Tensor] = None):
        """
        x: [B, T, input_dim|hidden_dim|temporal_latent_dim]

        We first apply a width-direction FFW to update everything but the
        input_dim, then apply some temporal processing along time.
        """

        # Only update the non-input dims
        x_update = self.ffw(self.norm1(x))
        x = F.pad(x_update, (self.in_dim, 0)) + x

        # Temporal attention on the trailing temporal_latent_dim slice
        t = x[..., -self.t_dim :]
        attn_mask = None
        if padding_mask is not None:
            # SDPA boolean mask: True = attend. Shape broadcasts to [B,H,T,T].
            keep = ~padding_mask
            attn_mask = keep[:, None, None, :]
        attention_update = self.temporal(self.norm2(t), rope=rope, attn_mask=attn_mask)
        x = torch.cat(
            [x[..., : -self.t_dim], t + attention_update],
            dim=-1,
        )
        return x


class SimpleBOLDEncoder(nn.Module):
    def __init__(
        self,
        n_parcels,
        input_dim,
        hidden_dim,
        temporal_dim,
        ffw_dim,
        depth,
        num_heads,
        preprocessor_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if preprocessor_hidden_dim is None:
            preprocessor_hidden_dim = n_parcels * 4
        self.preprocessor = FFW(n_parcels, preprocessor_hidden_dim, input_dim)
        self.hidden_dim, self.t_dim = hidden_dim, temporal_dim

        # Learned token used to replace the input-dim embedding at masked
        # timesteps during pretraining. Inert when `mask` is not passed.
        self.mask_token = nn.Parameter(torch.zeros(input_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                SimpleBrainEncoderLayer(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    temporal_latent_dim=temporal_dim,
                    ffw_dim=ffw_dim,
                    num_heads=num_heads,
                )
                for _ in range(depth)
            ]
        )

        # 1D rope over the time axis. We pass 2*head_dim because RotaryEmbeddingCat
        # produces a width-`dim` embed in 1D (only one spatial axis contributes
        # bands), which apply_rot_embed_cat then chunks in half — so dim must be
        # 2*head_dim for the halves to match the per-head width. feat_shape=None
        # keeps cached bands so we can rebuild the embed for variable T.
        self.rope = RotaryEmbeddingCat(
            2 * (temporal_dim // num_heads), in_pixels=False, feat_shape=None
        )

    def forward(
        self,
        parcels,
        padding_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        parcels: [B, T, n_parcels]
        mask: [B, T] bool. True positions have their input-dim embedding
            replaced by the learned `mask_token` (MLM-style pretraining).

        Returns the full hidden state [B, T, input_dim + hidden_dim + temporal_dim].
        """
        emb = self.preprocessor(parcels)
        if mask is not None:
            emb = torch.where(mask.unsqueeze(-1), self.mask_token, emb)
        # Zero-init the hidden + temporal channels appended to the input dims
        x = F.pad(emb, (0, self.hidden_dim + self.t_dim))
        rope = self.rope.get_embed(shape=[x.shape[1]])
        for block in self.blocks:
            x = block(x, rope=rope, padding_mask=padding_mask)
        return x
