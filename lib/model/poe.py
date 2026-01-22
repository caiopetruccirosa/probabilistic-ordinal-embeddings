import torch
import torch.nn as nn

from lib.utils import init_layer

class ProbabilisticOrdinalEmbeddingLayer(nn.Module):
    def __init__(self, in_dim: int, embedding_dim: int, t_samples: int):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.t_samples = t_samples
        self.embedding_mean = nn.Sequential(
            nn.Linear(self.in_dim, self.embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.embedding_log_var = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim, eps=1e-3, affine=False),
        )
        for m in self.embedding_mean.modules():
            init_layer(m)
        for m in self.embedding_log_var.modules():
            init_layer(m)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,  torch.Tensor]:
        embeddings_mean    = self.embedding_mean(x)    # shape (batch_size, embedding_dim)
        embeddings_log_var = self.embedding_log_var(x) # shape (batch_size, embedding_dim)

        # for numerical stability
        embeddings_log_var = torch.clamp(embeddings_log_var, min=-10.0, max=10.0) # shape (batch_size, embedding_dim)
        embeddings_std     = torch.exp(0.5 * embeddings_log_var)                  # shape (batch_size, embedding_dim)

        # monte-carlo sampling (reparameterization trick) by gaussian sampling
        eps = torch.randn(
            size=(self.t_samples, *embeddings_mean.shape),
            device=embeddings_mean.device,
            dtype=embeddings_mean.dtype
        ) # shape (t_samples, batch_size, embedding_dim)

        sampled_embeddings = embeddings_mean.unsqueeze(dim=0) + embeddings_std.unsqueeze(0) * eps  # shape (t_samples, batch_size, embedding_dim)

        return embeddings_mean, embeddings_log_var, sampled_embeddings