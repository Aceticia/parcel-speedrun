"""Simple BOLD encoder: input MLP + CLS token + standard transformer.

Per window, time-step parcels are projected by a 2-layer MLP into `dim`, a
learnable CLS token is prepended, learned positional embeddings are added, and
a stack of standard pre-norm `nn.TransformerEncoderLayer`s mixes everything.
The CLS output is the per-window summary feature.
"""

import torch
import torch.nn as nn


class SimpleBOLDEncoder(nn.Module):
    def __init__(
        self,
        n_parcels,
        dim=384,
        ffw_dim=1536,
        depth=6,
        num_heads=6,
        preprocessor_hidden_dim=512,
        max_len=64,
    ):
        super().__init__()
        self.preprocessor = nn.Sequential(
            nn.Linear(n_parcels, preprocessor_hidden_dim),
            nn.GELU(),
            nn.Linear(preprocessor_hidden_dim, dim),
        )
        self.mask_token = nn.Parameter(torch.empty(dim).normal_(std=0.02))
        self.cls_token = nn.Parameter(torch.empty(dim).normal_(std=0.02))
        self.pos_emb = nn.Parameter(torch.empty(max_len + 1, dim).normal_(std=0.02))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffw_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, parcels, mask=None):
        """parcels: [B, T, n_parcels]. mask: [B, T] bool — masked positions get
        the learned mask_token in place of the embedding. Returns [B, T+1, dim]
        where index 0 is the CLS token output and 1..T+1 are time-token outputs."""
        emb = self.preprocessor(parcels)
        if mask is not None:
            emb = torch.where(mask.unsqueeze(-1), self.mask_token, emb)
        B, T, _ = emb.shape
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, emb], dim=1) + self.pos_emb[: T + 1]
        return self.transformer(x)
