import math
import torch
import torch.nn as nn


class TFFLayer(nn.Module):
    def __init__(self, dim: int, local_hidden: int = 128, freq_list=(1, 2, 4, 8, 16)):
        super().__init__()
        self.dim = dim
        self.freq_list = freq_list
        self.local_gen = nn.Sequential(
            nn.Linear(dim, local_hidden),
            nn.GELU(),
            nn.Linear(local_hidden, 3 * dim)
        )

        self.mem_gate = nn.Linear(dim, dim)
        self.blend_fc = nn.Linear(dim, 3)
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, dim))

        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim * 2)  # will be split for GLU
        )
        self.norm_fiber = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

    def choose_freq_list(self, F, max_k=6):
        freqs = []
        k = 0
        while len(freqs) < max_k and (1 << k) <= F // 2:
            freqs.append(1 << k)
            k += 1
        return freqs or [1]

    def _memory_path(self, x_fiber: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.mem_gate(x_fiber))
        one_minus_alpha = 1.0 - alpha
        prefix_prod = torch.cumprod(alpha.flip(1), dim=1).flip(1)
        prefix_prod = torch.cat([torch.ones_like(prefix_prod[:, :1]), prefix_prod[:, :-1]], dim=1)
        m = torch.cumsum(one_minus_alpha * x_fiber * prefix_prod, dim=1)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.to(torch.bfloat16)
        if x.dim() == 4:
            B, F, P, D = x.shape
        elif x.dim() == 3:
            F, P, D = x.shape
            B = 1
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            P, D = x.shape
            B = 1
            F = 1
            x = x.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        x_fiber = x.permute(0, 2, 1, 3).reshape(B * P, F, D)
        fiber_mean = x_fiber.mean(dim=1)                     

        k = self.local_gen(fiber_mean).view(B * P, 3, D)     
        x_pad = torch.cat([x_fiber[:, -1:], x_fiber, x_fiber[:, :1]], dim=1)
        L = (
            k[:, 0].unsqueeze(1) * x_pad[:, 0:-2] +
            k[:, 1].unsqueeze(1) * x_pad[:, 1:-1] +
            k[:, 2].unsqueeze(1) * x_pad[:, 2:]
        )

        M = self._memory_path(x_fiber)
        if len(self.freq_list):
            t = torch.arange(F, device=x.device).float()                 
            P_acc = torch.zeros_like(x_fiber)
            for w in self.freq_list:
                phase = (2 * math.pi * w / F) * t                       
                cos = torch.cos(phase)[None, :, None]
                P_acc += cos * x_fiber
        else:
            P_acc = torch.zeros_like(x_fiber)

        beta = torch.softmax(self.blend_fc(fiber_mean), dim=-1)
        beta = beta.unsqueeze(1)
        beta_exp = beta.unsqueeze(-1)
        Y_fiber = (
            beta_exp[:, :, 0] * L +
            beta_exp[:, :, 1] * M +
            beta_exp[:, :, 2] * P_acc
        )

        Y = Y_fiber.reshape(B, P, F, D).permute(0, 2, 1, 3)
        c = Y.mean(dim=2, keepdim=True)                                
        Y = Y + self.gamma * c                                         

        out = x + self.norm_fiber(Y)
        ffn_in = self.norm_ffn(out)
        ffn_out = self.ffn(ffn_in)
        a, b = ffn_out.chunk(2, dim=-1)                   
        ffn_out = a * torch.sigmoid(b)
        out = out + ffn_out
        return out.squeeze()


class TemporalFiberFusion(nn.Module):
    """Stacked TFF layers.
    Args:
        dim (int): feature dimension *D*.
        depth (int): number of stacked TFFLayer blocks.
        local_hidden (int): hidden size for local branch kernel generator.
        freq_list (tuple[int]): frequencies for periodic branch.
    """

    def __init__(self, dim: int = 1152, depth: int = 3,
                 local_hidden: int = 1152, freq_list=(1, 2, 4, 8, 16)):
        super().__init__()
        self.layers = nn.ModuleList([
            TFFLayer(dim, local_hidden, freq_list) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through *depth* TFF layers."""
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    B, F, P, D = 2, 128, 256, 1152
    dummy = torch.randn(B, F, P, D)

    tff = TemporalFiberFusion(dim=D, depth=3)
    out = tff(dummy)
    print("Output shape:", out.shape)  # should be [B, F, P, D]
