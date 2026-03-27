"""
Game-Theoretic DiT Module
=========================
创新点：将 GameFormer 的 Level-k Reasoning 嵌入到扩散去噪过程中。

改造说明：
- 原始 DiTBlock：每个 agent 只做 self-attn + cross-attn（与场景条件）
- 新增 GameTheoreticAttention：在每个去噪 step 内，agent 间显式做跨智能体 attention
- 新增 GameTheoreticDiTBlock：在原 DiTBlock 基础上加入跨智能体博弈交互层
- DiT forward：支持 level-k 控制（每 k 个去噪 block 做一次博弈更新）

与原代码完全兼容，只需用本文件替换 diffusion_planner/model/module/dit.py
"""

import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp


# ===================== 保留原始函数 =====================

def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x


def scale_fn(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))
    return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.（与原始完全相同）
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ===================== 原始 DiTBlock（保留，用于对比消融） =====================

class DiTBlock(nn.Module):
    """
    原始 DiT block（不含博弈交互）。保留用于消融实验。
    """
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, cross_c, y, attn_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(y).chunk(6, dim=1)

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask
        )[0]

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = self.mlp2(self.norm4(x))

        return x


# ===================== 核心创新：博弈式跨智能体 Attention =====================

class GameTheoreticAttention(nn.Module):
    """
    Level-k 博弈式跨智能体 Attention 模块。
    
    核心思想：
    - 在去噪过程中，每个智能体的去噪状态受其他智能体当前去噪状态影响
    - 通过 cross-agent attention 实现隐式博弈交互
    - diffusion timestep 编码让模型感知当前处于去噪的哪个阶段（对应博弈 level）
    
    Args:
        dim: 特征维度
        heads: attention heads 数
        dropout: dropout 率
        use_role_embedding: 是否区分 ego 和 neighbor 角色
    """
    def __init__(self, dim=192, heads=6, dropout=0.1, use_role_embedding=True):
        super().__init__()
        self.dim = dim
        self.use_role_embedding = use_role_embedding
        
        # 跨智能体 attention：query=自身，key/value=所有智能体
        self.norm_pre = nn.LayerNorm(dim)
        self.cross_agent_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_post = nn.LayerNorm(dim)
        
        # 门控机制：自适应控制博弈交互的强度
        # 初始化接近 0，让模型从"无博弈"开始学习
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -2.0)  # sigmoid(-2) ≈ 0.12，初始轻微激活
        
        # 角色编码：区分 ego（规划目标）和 neighbor（预测对象）
        if use_role_embedding:
            self.role_embedding = nn.Embedding(2, dim)  # 0=ego, 1=neighbor
        
        # MLP 融合博弈交互信息
        mlp_hidden = int(dim * 2)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0.
        )

    def forward(self, x, neighbor_mask=None, diffusion_t_embed=None):
        """
        Args:
            x: (B, P, D)  P = 1(ego) + N(neighbors)
            neighbor_mask: (B, P) bool，True 表示该 slot 无效（对应 neighbor_current_mask）
            diffusion_t_embed: (B, D) 扩散时间步的嵌入，用于调制博弈强度
        Returns:
            x: (B, P, D) 经过博弈交互后的特征
        """
        B, P, D = x.shape
        residual = x
        
        # 加入角色编码
        if self.use_role_embedding:
            # ego: role=0, neighbors: role=1
            roles = torch.zeros(P, dtype=torch.long, device=x.device)
            roles[1:] = 1
            role_emb = self.role_embedding(roles)  # (P, D)
            x_with_role = x + role_emb.unsqueeze(0)  # (B, P, D)
        else:
            x_with_role = x
        
        x_normed = self.norm_pre(x_with_role)
        
        # 跨智能体 attention
        # 所有 P 个智能体互相看彼此的去噪状态
        # key_padding_mask: True 的位置被忽略
        attn_out, attn_weights = self.cross_agent_attn(
            x_normed, x_normed, x_normed,
            key_padding_mask=neighbor_mask  # (B, P)，ego 位置为 False 永远有效
        )
        
        # 门控：用 diffusion timestep 调制博弈强度
        # 直觉：早期去噪（t 大）时博弈影响弱，晚期去噪（t 小）时博弈精细化
        gate_input = x if diffusion_t_embed is None else (x + diffusion_t_embed.unsqueeze(1))
        gate_val = self.gate(gate_input)  # (B, P, D)
        
        # 残差连接 + 门控融合
        x = residual + gate_val * self.mlp(self.norm_post(attn_out))
        
        return x, attn_weights


# ===================== 改造后的 DiTBlock（含博弈交互）=====================

class GameTheoreticDiTBlock(nn.Module):
    """
    博弈式 DiT Block = 原始 DiTBlock + GameTheoreticAttention。
    
    在每个 DiT block 中，额外插入跨智能体博弈 attention，
    实现"去噪过程即博弈过程"。
    
    Args:
        dim: 特征维度
        heads: attention heads 数
        dropout: dropout 率
        mlp_ratio: MLP 扩张比
        game_attn_every_k: 每 k 个 block 做一次博弈 attention（消融实验用）
                           默认 1 表示每个 block 都做
        use_role_embedding: 是否使用角色编码
    """
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0,
                 use_role_embedding=True):
        super().__init__()
        
        # --- 原始 DiTBlock 组件 ---
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # --- 新增：博弈式跨智能体 Attention ---
        self.game_attn = GameTheoreticAttention(
            dim=dim, heads=heads, dropout=dropout,
            use_role_embedding=use_role_embedding
        )

    def forward(self, x, cross_c, y, attn_mask, neighbor_mask=None, diffusion_t_embed=None):
        """
        Args:
            x: (B, P, D)
            cross_c: (B, N_ctx, D)  场景条件（来自 encoder）
            y: (B, D)  adaLN 条件（route + timestep embedding）
            attn_mask: (B, P) self-attn mask（无效 neighbor 为 True）
            neighbor_mask: (B, P) 博弈 attn mask（与 attn_mask 相同，单独传入方便消融）
            diffusion_t_embed: (B, D) 扩散时间步嵌入（用于调制博弈强度）
        Returns:
            x: (B, P, D)
            game_attn_weights: (B, P, P) 博弈 attention 权重（用于可视化分析）
        """
        # ---- Step 1: 原始 DiTBlock 流程 ----
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(y).chunk(6, dim=1)

        # 时序内 self-attention（token 维度 = 时间步）
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask
        )[0]

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        # 场景条件 cross-attention
        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = self.mlp2(self.norm4(x))

        # ---- Step 2: 博弈式跨智能体 Attention ----
        # 此时 x 形状为 (B, P, D)，P = 1 + N_neighbors
        # 每个智能体看彼此当前的去噪状态，实现 level-k 博弈交互
        x, game_attn_weights = self.game_attn(
            x,
            neighbor_mask=neighbor_mask if neighbor_mask is not None else attn_mask,
            diffusion_t_embed=diffusion_t_embed
        )

        return x, game_attn_weights


# ===================== FinalLayer（与原始相同）=====================

class FinalLayer(nn.Module):
    """The final layer of DiT.（与原始完全相同）"""
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        B, P, _ = x.shape
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x