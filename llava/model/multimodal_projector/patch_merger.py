import torch
import torch.nn as nn
import math
from torch.nn import LayerNorm

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# class PatchMerger(nn.Module):
#     def __init__(self, config, vision_cfg, spatial_merge_size: int = 2, temporal_merge_size: int = 2):
#         super().__init__()
#         self.spatial_merge_size = spatial_merge_size
#         self.temporal_merge_size = temporal_merge_size
#         self.hidden_size = config.mm_hidden_size * (spatial_merge_size**2) * temporal_merge_size
        
#         self.ln_q = RMSNorm(self.hidden_size, eps=1e-6)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.GELU(),
#             nn.Linear(self.hidden_size, config.hidden_size),
#         )

#     def forward(self, x, window_index=None):
#         """
#         x: Tensor of shape (T, P, C) where:
#             T = 帧数（视频: 多帧, 图像: 1帧）
#             P = patch 数（空间划分）
#             C = 特征维度
#         """
#         T, P, C = x.shape
        
#         # 处理单帧图像情况：复制成 2 帧以适配 temporal merge
#         if T == 1:
#             x = x.repeat(2, 1, 1)  # (2, P, C)
#             T = 2
        
#         # Temporal padding: 如果 T 不能被 temporal_merge_size 整除，则填充最后一帧
#         if T % self.temporal_merge_size != 0:
#             pad_size = self.temporal_merge_size - (T % self.temporal_merge_size)
#             pad_frames = x[-1:].repeat(pad_size, 1, 1)
#             x = torch.cat([x, pad_frames], dim=0)
#             T += pad_size
        
#         # Reshape: 进行时间 + 空间 merge
#         new_T = T // self.temporal_merge_size
#         new_P = P // (self.spatial_merge_size**2)
#         new_C = C * (self.spatial_merge_size**2) * self.temporal_merge_size
        
#         x = x.view(new_T, self.temporal_merge_size, new_P, self.spatial_merge_size**2, C)
#         x = x.permute(0, 2, 1, 3, 4).reshape(new_T, new_P, new_C)
        
#         x = self.mlp(self.ln_q(x))
#         if window_index is not None:
#             reverse_indices = torch.argsort(window_index)
#             x = x[reverse_indices, :]
        
#         return x

#     @property
#     def config(self):
#         return {"mm_projector_type": "patchmerger"}


class PatchMerger(nn.Module):
    def __init__(self, config, vision_cfg, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = config.mm_hidden_size * (spatial_merge_size**2)
        self.ln_q = RMSNorm(config.mm_hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.hidden_size),
        )

    def forward(self, x, window_index):
        x = self.mlp(self.ln_q(x).reshape(-1, self.hidden_size).to(torch.bfloat16))
        if window_index is not None:
            reverse_indices = torch.argsort(window_index)
            x = x[reverse_indices, :]
        return x

    @property
    def config(self):
        return {"mm_projector_type": "patchmerger"}