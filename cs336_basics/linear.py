import torch
import torch.nn as nn
import einops


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros(
                    out_features, in_features, device=self.device, dtype=self.dtype
                ),
                std=2 / (in_features + out_features),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(
            x,
            self.weights,
            "... in_features, out_features in_features -> ... out_features",
        )
