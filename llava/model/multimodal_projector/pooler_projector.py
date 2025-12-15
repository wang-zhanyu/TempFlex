import torch
import torch.nn as nn

import math

from transformers.models.clip.modeling_clip import CLIPVisionModel


class PoolerProjector(nn.Module):
    def __init__(self, config, vision_cfg):
        super().__init__()
        self._config = config
        self.hw = vision_cfg.image_size // vision_cfg.patch_size

        self.conv_pool = nn.Conv2d(config.mm_hidden_size, config.hidden_size, kernel_size=2, stride=2)

        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        height = width = self.hw
        assert height * width == x.shape[1]
        x = x.view(x.shape[0], height, width, -1).permute(0, 3, 1, 2)
        x = self.conv_pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

    @property
    def config(self):
        return {"mm_projector_type": "pooler"}


class PoolerProjectorV2(nn.Module):
    def __init__(self, config, vision_cfg):
        super().__init__()
        self._config = config
        self.hw = vision_cfg.image_size // vision_cfg.patch_size  # e.g. 27
        # mm_hidden_size=1152, hidden_size=4096 in your case
        
        # -------------- 卷积池化部分 (与之前相同) --------------
        self.conv_pre = nn.Sequential(
            nn.Conv2d(config.mm_hidden_size, config.mm_hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config.mm_hidden_size),
            nn.GELU(),
        )
        self.conv_down = nn.Conv2d(config.mm_hidden_size, config.hidden_size, kernel_size=2, stride=2, padding=1)
        self.bn_down = nn.BatchNorm2d(config.hidden_size)
        
        # 残差分支 (1x1 conv, stride=2) 将通道从 mm_hidden_size -> hidden_size
        self.res_conv = nn.Conv2d(config.mm_hidden_size, config.hidden_size, kernel_size=1, stride=2, padding=0)

        # -------------- 位置编码 --------------
        # 下采样后空间变为 hw//2 x hw//2
        self.out_hw = self.hw // 2  # e.g. 13
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (self.out_hw+1) * (self.out_hw+1), config.hidden_size)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # -------------- 一层 FFN + 残差 --------------
        # 只保留单层线性映射 (hidden_size -> hidden_size)
        # 加上LayerNorm和Dropout, 并做残差
        self.ln = nn.LayerNorm(config.hidden_size)
        self.proj_fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.drop = nn.Dropout(p=0.1)  # 可根据需求调整

    def forward(self, x):
        """
        x: (B, N=hw*hw, C_in=mm_hidden_size)
        return: (B, out_hw^2, hidden_size)
        """
        B, N, C = x.shape
        height = width = self.hw
        assert height * width == N, f"Expected {height}*{width}={N}, got {N}"

        # 1) Reshape => (B, C_in, 27, 27)
        x_2d = x.view(B, height, width, C).permute(0, 3, 1, 2)

        # 2) conv_pre => (B, C_in, 27, 27)
        x_pre = self.conv_pre(x_2d)

        # 3) conv_down => (B, hidden_size, 13, 13), BN + GELU
        x_down = self.conv_down(x_pre)
        x_down = self.bn_down(x_down)
        x_down = nn.functional.gelu(x_down)

        # 4) 残差支路 => (B, hidden_size, 13, 13)
        x_res = self.res_conv(x_2d)

        # 5) 相加 => (B, hidden_size, 13, 13)
        x_2d_out = x_down + x_res

        # 6) flatten => (B, 169, hidden_size) 其中169=13*13
        x_seq = x_2d_out.flatten(2).transpose(1, 2)  # (B, out_hw^2, hidden_size)

        # 7) 加 position embedding => (B, 169, hidden_size)
        x_seq = x_seq + self.pos_embed

        # 8) 一层 FFN + 残差
        #    LN -> proj_fc -> Dropout -> Residual
        x_res2 = x_seq
        x_seq = self.ln(x_seq)                # (B, 169, 4096)
        x_seq = self.proj_fc(x_seq)           # (B, 169, 4096)
        x_seq = self.drop(x_seq)              # (B, 169, 4096)
        x_seq = x_res2 + x_seq                # 残差

        return x_seq  # (B, 169, 4096)

    @property
    def config(self):
        return {"mm_projector_type": "pooler_v2"}