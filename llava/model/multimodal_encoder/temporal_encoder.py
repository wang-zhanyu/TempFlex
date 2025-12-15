import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def activation(act):
    if act == 'RELU':
        return nn.ReLU()
    elif act == 'TANH':
        return nn.Tanh()
    elif act == 'GLU':
        return nn.GLU()
    elif act == 'ELU':
        return nn.ELU(1.3)
    elif act == 'CELU':
        return nn.CELU(1.3)
    else:
        return nn.Identity()


class Encoder(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        dropout, 
        att_heads, 
        att_mid_dim, 
        att_mid_drop, 
        bifeat_emb_act, 
        bifeat_emb_drop, 
        ff_dropout, 
        layer_num
    ):
        super(Encoder, self).__init__()
        self.att_heads = att_heads
        self.layers = nn.ModuleList([])     
        for i in range(layer_num):
            sublayer = EncoderLayer( 
                embed_dim = embed_dim, 
                dropout = dropout, 
                att_heads = att_heads, 
                att_mid_dim = att_mid_dim, 
                att_mid_drop = att_mid_drop, 
                bifeat_emb_act = bifeat_emb_act, 
                bifeat_emb_drop = bifeat_emb_drop, 
                ff_dropout = ff_dropout)
            self.layers.append(sublayer)
        
        self.proj_norm = nn.Sequential(
            nn.Linear(embed_dim * (layer_num + 1), embed_dim), 
            torch.nn.LayerNorm(embed_dim))

    def forward(self, x, mask=None):
        x = x.to(torch.bfloat16)
        if mask is not None:
            gx = (torch.sum(x * mask.unsqueeze(-1), 1) / torch.sum(mask.unsqueeze(-1), 1))
        else:
            gx = torch.mean(x, 1)
        gx_arr = [gx]
        for layer in self.layers:
            gx, x = layer(gx, x, mask)
            gx_arr.append(gx)

        gx = torch.cat(gx_arr, dim=-1)
        gx = self.proj_norm(gx)
        to_return = {'last_hidden_state':x, 'pooler_output':gx}
        return to_return


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, 
        relu_dropout, dropout):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.layer_norms = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.layer_norms(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        dropout, 
        att_heads, 
        att_mid_dim, 
        att_mid_drop,
        bifeat_emb_act, 
        bifeat_emb_drop, 
        ff_dropout
    ):
        super(EncoderLayer, self).__init__()
        self.encoder_attn = LowRank(
            embed_dim = embed_dim, 
            att_heads = att_heads, 
            att_mid_dim = att_mid_dim, 
            att_mid_drop = att_mid_drop)
        self.dropout = nn.Dropout(dropout)

        self.bifeat_emb = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            activation(bifeat_emb_act),
            nn.Dropout(bifeat_emb_drop)
        )
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        self.ff_layer = FeedForwardBlock(
            embed_dim = embed_dim, 
            ffn_embed_dim = embed_dim * 4, 
            relu_dropout = ff_dropout, 
            dropout = ff_dropout)

    def forward(self, gx, x, mask):
        gx = gx.to(torch.bfloat16)
        x = x.to(torch.bfloat16)

        gx = self.encoder_attn(
            query = gx,
            key = x,
            mask = mask,
            value1 = gx,
            value2 = x
        )
        gx = self.dropout(gx)

        x_ = torch.cat([gx.unsqueeze(1).expand_as(x), x], dim = -1)
        x = self.bifeat_emb(x_) + x
        x = self.layer_norm(x)

        if self.ff_layer is not None:
            x = self.ff_layer(x)
        return gx, x


class LowRank(nn.Module):
    def __init__(self, embed_dim, att_heads, att_mid_dim, att_mid_drop, act_type="CELU"):
        super(LowRank, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = att_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        output_dim = 2 * embed_dim if act_type == 'GLU' else embed_dim

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act_type)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_q = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act_type)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_k = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act_type)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v1 = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = activation(act_type)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.in_proj_v2 = nn.Sequential(*sequential)

        self.attn_net = SCAtt(att_mid_dim, att_mid_drop)
        self.clear_buffer() 

    def apply_to_states(self, fn):
        self.buffer_keys = fn(self.buffer_keys)
        self.buffer_value2 = fn(self.buffer_value2)

    def init_buffer(self, batch_size):
        self.buffer_keys = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()
        self.buffer_value2 = torch.zeros((batch_size, self.num_heads, 0, self.head_dim)).cuda()

    def clear_buffer(self):
        self.buffer_keys = None
        self.buffer_value2 = None

    # query -- batch_size * qdim
    # value -- batch_size * att_num * vdim
    def forward(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value1 = value1.to(torch.bfloat16)
        value2 = value2.to(torch.bfloat16)

        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim)

        if precompute == False:
            key = key.reshape(-1, key.size()[-1])
            value2 = value2.reshape(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = key
            v2 = value2

        attn_map = q.unsqueeze(-2) * k
        attn = self.attn_net(attn_map, mask, v1, v2)
        attn = attn.view(batch_size, self.num_heads * self.head_dim)
        return attn

    # query -- batch_size * seq_num * qdim
    # value -- batch_size * att_num * vdim
    def forward2(self, query, key, mask, value1, value2, precompute=False):
        batch_size = query.size()[0]
        query = query.view(-1, query.size()[-1])
        value1 = value1.view(-1, value1.size()[-1])
        
        q = self.in_proj_q(query)
        v1 = self.in_proj_v1(value1)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v1 = v1.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if precompute == False:
            key = key.view(-1, key.size()[-1])
            value2 = value2.view(-1, value2.size()[-1])
            k = self.in_proj_k(key)
            v2 = self.in_proj_v2(value2)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            if self.buffer_keys is not None and self.buffer_value2 is not None:
                self.buffer_keys = torch.cat([self.buffer_keys, k], dim=2)
                self.buffer_value2 = torch.cat([self.buffer_value2, v2], dim=2)
                k = self.buffer_keys
                v2 = self.buffer_value2
        else:
            k = key
            v2 = value2
        
        attn_map = q.unsqueeze(-2) * k.unsqueeze(-3)
        attn = self.attn_net.forward(attn_map, mask, v1, v2).transpose(1, 2).contiguous()
        attn = attn.view(batch_size, -1, self.num_heads * self.head_dim)
        return attn

    def precompute(self, key, value2):
        batch_size = value2.size()[0]
        key = key.view(-1, key.size()[-1])
        value2 = value2.view(-1, value2.size()[-1])

        k = self.in_proj_k(key)
        v2 = self.in_proj_v2(value2)

        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        return k, v2


class BasicAtt(nn.Module):
    def __init__(self, mid_dims, mid_dropout):
        super(BasicAtt, self).__init__()

        sequential = []
        for i in range(1, len(mid_dims) - 1):
            sequential.append(nn.Linear(mid_dims[i - 1], mid_dims[i]))
            sequential.append(nn.ReLU())
            if mid_dropout > 0:
                sequential.append(nn.Dropout(mid_dropout))
        self.attention_basic = nn.Sequential(*sequential) if len(sequential) > 0 else None
        self.attention_last = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)
        attn_weights = self.attention_last(att_map)
        attn_weights = attn_weights.squeeze(-1)
        if att_mask is not None:
            attn_weights = attn_weights.masked_fill(att_mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn = torch.matmul(attn_weights.unsqueeze(-2), value2).squeeze(-2)
        return attn


class SCAtt(BasicAtt):
    def __init__(self, mid_dims, mid_dropout):
        super(SCAtt, self).__init__(mid_dims, mid_dropout)
        self.attention_last = nn.Linear(mid_dims[-2], 1)
        self.attention_last2 = nn.Linear(mid_dims[-2], mid_dims[-1])

    def forward(self, att_map, att_mask, value1, value2):
        if self.attention_basic is not None:
            att_map = self.attention_basic(att_map)

        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1)
            att_mask_ext = att_mask.unsqueeze(-1)
            att_map_pool = torch.sum(att_map * att_mask_ext, -2) / torch.sum(att_mask_ext, -2)
        else:
            att_map_pool = att_map.mean(-2)

        alpha_spatial = self.attention_last(att_map)
        alpha_channel = self.attention_last2(att_map_pool)
        alpha_channel = torch.sigmoid(alpha_channel)

        alpha_spatial = alpha_spatial.squeeze(-1)
        if att_mask is not None:
            alpha_spatial = alpha_spatial.masked_fill(att_mask == 0, np.half(-1e9))
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)

        if len(alpha_spatial.shape) == 4: # batch_size * head_num * seq_num * seq_num (for xtransformer)
            value2 = torch.matmul(alpha_spatial, value2)
        else:
            value2 = torch.matmul(alpha_spatial.unsqueeze(-2), value2).squeeze(-2)

        attn = value1 * value2 * alpha_channel
        return attn



class TemporalEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=1152,
        temporal_dropout=0.2,
        temporal_heads=8,
        temporal_mid_dim=[144,96,144],
        temporal_mid_drop=0.1,
        temporal_bifeat_act="RELU",
        temporal_bifeat_drop=0.3,
        temporal_ff_dropout=0.1,
        temporal_layers=2
    ):
        super().__init__()

        self.temporal_encoder = Encoder(
            embed_dim=embed_dim,
            dropout=temporal_dropout,
            att_heads=temporal_heads,
            att_mid_dim=temporal_mid_dim,
            att_mid_drop=temporal_mid_drop,
            bifeat_emb_act=temporal_bifeat_act,
            bifeat_emb_drop=temporal_bifeat_drop,
            ff_dropout=temporal_ff_dropout,
            layer_num=temporal_layers
        )
    
    def forward(self, x, temporal_mask=None):
        if x.dim() == 4:
            B, F, P, D = x.shape
        elif x.dim() == 3:
            F, P, D = x.shape
            B = 1
            x = x.unsqueeze(0)  # -> [1, F, P, D]
        elif x.dim() == 2:
            P, D = x.shape
            B = 1
            F = 1
            x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, P, D]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        x_pf = x.permute(0, 2, 1, 3)  # [B, P, F, D]
        x_pf = x_pf.reshape(B * P, F, D)
        if temporal_mask is not None:
            m_pf = temporal_mask.unsqueeze(1).expand(-1, P, -1)  # [B, P, F]
            m_pf = m_pf.reshape(B * P, F)
        else:
            m_pf = None

        out_pf = self.temporal_encoder(x_pf, m_pf)
        x_temporal = out_pf['last_hidden_state']
        x_out = x_temporal.reshape(B, P, F, D).permute(0, 2, 1, 3)
        return x_out.squeeze()



class SpatialTemporalEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=1152,
        spatial_dropout=0.5,
        spatial_heads=8,
        spatial_mid_dim=[144,96,144],
        spatial_mid_drop=0.1,
        spatial_bifeat_act="RELU",
        spatial_bifeat_drop=0.3,
        spatial_ff_dropout=0.1,
        spatial_layers=2,
        temporal_dropout=0.2,
        temporal_heads=8,
        temporal_mid_dim=[144,96,144],
        temporal_mid_drop=0.1,
        temporal_bifeat_act="RELU",
        temporal_bifeat_drop=0.3,
        temporal_ff_dropout=0.1,
        temporal_layers=2
    ):
        super().__init__()

        self.spatial_encoder = Encoder(
            embed_dim=embed_dim, 
            dropout=spatial_dropout,
            att_heads=spatial_heads,
            att_mid_dim=spatial_mid_dim,
            att_mid_drop=spatial_mid_drop,
            bifeat_emb_act=spatial_bifeat_act,
            bifeat_emb_drop=spatial_bifeat_drop,
            ff_dropout=spatial_ff_dropout,
            layer_num=spatial_layers
        )

        self.temporal_encoder = Encoder(
            embed_dim=embed_dim,
            dropout=temporal_dropout,
            att_heads=temporal_heads,
            att_mid_dim=temporal_mid_dim,
            att_mid_drop=temporal_mid_drop,
            bifeat_emb_act=temporal_bifeat_act,
            bifeat_emb_drop=temporal_bifeat_drop,
            ff_dropout=temporal_ff_dropout,
            layer_num=temporal_layers
        )
    
    def forward(self, x, spatial_mask=None, temporal_mask=None):
        if x.dim() == 4:
            B, F, P, D = x.shape
        elif x.dim() == 3:
            F, P, D = x.shape
            B = 1
            x = x.unsqueeze(0)  # -> [1, F, P, D]
        elif x.dim() == 2:
            P, D = x.shape
            B = 1
            F = 1
            x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, P, D]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        x_sf = x.reshape(B * F, P, D)
        if spatial_mask is not None:
            m_sf = spatial_mask.reshape(B * F, P)
        else:
            m_sf = None

        out_sf = self.spatial_encoder(x_sf, m_sf)
        x_spatial = out_sf['last_hidden_state']

        x_pf = x_spatial.reshape(B, F, P, D).permute(0, 2, 1, 3)  # [B, P, F, D]
        x_pf = x_pf.reshape(B * P, F, D)
        if temporal_mask is not None:
            m_pf = temporal_mask.unsqueeze(1).expand(-1, P, -1)  # [B, P, F]
            m_pf = m_pf.reshape(B * P, F)
        else:
            m_pf = None

        out_pf = self.temporal_encoder(x_pf, m_pf)
        x_temporal = out_pf['last_hidden_state']
        x_out = x_temporal.reshape(B, P, F, D).permute(0, 2, 1, 3)
        return x_out.squeeze()


if __name__ == "__main__":
    # encoder = Encoder(
    #         embed_dim = 768, 
    #         dropout = 0.5, 
    #         att_heads=8, 
    #         att_mid_dim=[96, 64, 96], 
    #         att_mid_drop=0.1, 
    #         bifeat_emb_act="RELU", 
    #         bifeat_emb_drop=0.3, 
    #         ff_dropout=0.1, 
    #         layer_num=4)

    # encoder = encoder.to('cuda')
    # print(out['last_hidden_state'].shape)

    # video_encoder = SpatialTemporalEncoder()
    # video_encoder = video_encoder.to('cuda')
    # x = torch.rand(1, 128, 256, 768).cuda()
    # mask = torch.ones(1, 128, 256).cuda()
    # out = video_encoder(x)
    # print(out.shape)


    video_encoder = TemporalEncoder()
    video_encoder = video_encoder.to('cuda')
    x = torch.rand(1, 128, 256, 1152).cuda()
    mask = torch.ones(1, 128, 256).cuda()
    out = video_encoder(x)
    print(out.shape)
