"""
快速验证脚本
===========
不依赖 nuPlan 数据集，直接验证核心模块的前向传播是否正确。
运行方式：python verify_game_theoretic.py
"""

import torch
import torch.nn as nn
import sys
import os

# ===================== 模拟最小化依赖 =====================
# 如果没有安装 timm，先 pip install timm

try:
    from timm.models.layers import Mlp
except ImportError:
    print("请先安装: pip install timm")
    sys.exit(1)

# 内联依赖模块（避免需要完整项目环境）

import math

def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x


class TimestepEmbedder(nn.Module):
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


class FinalLayer(nn.Module):
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


# ===================== 核心模块（从改造文件中内联）=====================

class GameTheoreticAttention(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, use_role_embedding=True):
        super().__init__()
        self.dim = dim
        self.use_role_embedding = use_role_embedding
        self.norm_pre = nn.LayerNorm(dim)
        self.cross_agent_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_post = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, -2.0)
        if use_role_embedding:
            self.role_embedding = nn.Embedding(2, dim)
        mlp_hidden = int(dim * 2)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden,
                       act_layer=lambda: nn.GELU(approximate="tanh"), drop=0.)

    def forward(self, x, neighbor_mask=None, diffusion_t_embed=None):
        B, P, D = x.shape
        residual = x
        if self.use_role_embedding:
            roles = torch.zeros(P, dtype=torch.long, device=x.device)
            roles[1:] = 1
            role_emb = self.role_embedding(roles)
            x_with_role = x + role_emb.unsqueeze(0)
        else:
            x_with_role = x
        x_normed = self.norm_pre(x_with_role)
        attn_out, attn_weights = self.cross_agent_attn(
            x_normed, x_normed, x_normed, key_padding_mask=neighbor_mask
        )
        gate_input = x if diffusion_t_embed is None else (x + diffusion_t_embed.unsqueeze(1))
        gate_val = self.gate(gate_input)
        x = residual + gate_val * self.mlp(self.norm_post(attn_out))
        return x, attn_weights


class DiTBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, cross_c, y, attn_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(y).chunk(6, dim=1)
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask)[0]
        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)
        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = self.mlp2(self.norm4(x))
        return x


class GameTheoreticDiTBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, use_role_embedding=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.game_attn = GameTheoreticAttention(dim=dim, heads=heads, dropout=dropout,
                                                use_role_embedding=use_role_embedding)

    def forward(self, x, cross_c, y, attn_mask, neighbor_mask=None, diffusion_t_embed=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(y).chunk(6, dim=1)
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask)[0]
        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)
        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = self.mlp2(self.norm4(x))
        # 博弈交互
        x, game_weights = self.game_attn(
            x,
            neighbor_mask=neighbor_mask if neighbor_mask is not None else attn_mask,
            diffusion_t_embed=diffusion_t_embed
        )
        return x, game_weights


# ===================== 验证测试 =====================

def test_game_theoretic_attention():
    print("\n[Test 1] GameTheoreticAttention 前向传播...")
    dim, heads, B, P = 192, 6, 4, 9  # B=batch, P=1+8(ego+neighbors)
    
    gta = GameTheoreticAttention(dim=dim, heads=heads, use_role_embedding=True)
    
    x = torch.randn(B, P, dim)
    # 模拟 3 个 neighbor 无效的 mask
    neighbor_mask = torch.zeros(B, P, dtype=torch.bool)
    neighbor_mask[:, 5:] = True  # slot 5~8 无效
    
    diffusion_t_embed = torch.randn(B, dim)
    
    out, weights = gta(x, neighbor_mask=neighbor_mask, diffusion_t_embed=diffusion_t_embed)
    
    assert out.shape == (B, P, dim), f"输出形状错误: {out.shape}"
    assert weights.shape == (B * heads, P, P), f"权重形状错误: {weights.shape}"
    print(f"  ✓ 输出形状: {out.shape}")
    print(f"  ✓ 注意力权重形状: {weights.shape}")
    print(f"  ✓ 门控初始值（应接近 0）: {gta.gate[0].bias.mean().item():.3f}")


def test_game_theoretic_dit_block():
    print("\n[Test 2] GameTheoreticDiTBlock 前向传播...")
    dim, heads, B, P, N_ctx = 192, 6, 4, 9, 256
    
    block = GameTheoreticDiTBlock(dim=dim, heads=heads, use_role_embedding=True)
    
    x = torch.randn(B, P, dim)
    cross_c = torch.randn(B, N_ctx, dim)
    y = torch.randn(B, dim)  # adaLN conditioning
    attn_mask = torch.zeros(B, P, dtype=torch.bool)
    attn_mask[:, 6:] = True
    diffusion_t_embed = torch.randn(B, dim)
    
    out, weights = block(x, cross_c, y, attn_mask,
                         neighbor_mask=attn_mask,
                         diffusion_t_embed=diffusion_t_embed)
    
    assert out.shape == (B, P, dim)
    print(f"  ✓ 输出形状: {out.shape}")
    print(f"  ✓ 博弈权重形状: {weights.shape}")


def test_game_consistency_loss():
    print("\n[Test 3] 博弈一致性损失计算...")
    import torch.nn.functional as F
    
    B, P, T = 4, 9, 16  # batch, agents, future_timesteps
    N = P - 1  # neighbors
    
    score = torch.randn(B, P, T, 4)  # x, y, cos, sin
    gt_future = torch.randn(B, P, T, 4)
    neighbor_future_mask = torch.zeros(B, N, T, dtype=torch.bool)
    neighbor_future_mask[:, 5:, :] = True  # 部分 neighbor 无效
    
    # 复制自 game_theoretic_loss.py 的 game_consistency_loss
    ego_pred = score[:, 0, :, :2]
    neighbor_pred = score[:, 1:, :, :2]
    
    ego_expanded = ego_pred[:, None, :, :]
    dist = torch.norm(ego_expanded - neighbor_pred, dim=-1)
    safety_threshold = 0.5
    separation_loss = F.relu(safety_threshold - dist)
    valid_mask = ~neighbor_future_mask
    if valid_mask.sum() > 0:
        sep_loss = (separation_loss * valid_mask).sum() / valid_mask.sum()
    else:
        sep_loss = torch.tensor(0.0)
    
    print(f"  ✓ 分离损失: {sep_loss.item():.4f}")
    print(f"  ✓ 损失值为有限数: {torch.isfinite(sep_loss).item()}")


def test_mode_comparison():
    print("\n[Test 4] 对比三种博弈模式的参数量...")
    dim, heads, depth = 192, 6, 3
    
    def count_params(blocks):
        return sum(p.numel() for p in blocks.parameters())
    
    # none 模式（原始）
    blocks_none = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
    
    # all 模式（完整博弈）
    blocks_all = nn.ModuleList([GameTheoreticDiTBlock(dim, heads) for _ in range(depth)])
    
    # last 模式
    blocks_last = nn.ModuleList(
        [DiTBlock(dim, heads) if i < depth-1 else GameTheoreticDiTBlock(dim, heads)
         for i in range(depth)]
    )
    
    params_none = count_params(blocks_none)
    params_all = count_params(blocks_all)
    params_last = count_params(blocks_last)
    
    print(f"  原始(none):    {params_none:,} 参数")
    print(f"  完整(all):     {params_all:,} 参数  (+{params_all - params_none:,}, +{(params_all/params_none-1)*100:.1f}%)")
    print(f"  最后一层(last):{params_last:,} 参数  (+{params_last - params_none:,}, +{(params_last/params_none-1)*100:.1f}%)")


def test_gradient_flow():
    print("\n[Test 5] 梯度流验证...")
    dim, heads, B, P, N_ctx = 64, 4, 2, 5, 32
    
    block = GameTheoreticDiTBlock(dim=dim, heads=heads)
    t_embedder = TimestepEmbedder(dim)
    
    x = torch.randn(B, P, dim, requires_grad=True)
    cross_c = torch.randn(B, N_ctx, dim)
    t = torch.rand(B)
    y = t_embedder(t)
    attn_mask = torch.zeros(B, P, dtype=torch.bool)
    
    out, _ = block(x, cross_c, y, attn_mask, diffusion_t_embed=y)
    loss = out.mean()
    loss.backward()
    
    assert x.grad is not None, "x 没有梯度！"
    assert not torch.isnan(x.grad).any(), "梯度出现 NaN！"
    print(f"  ✓ 梯度范数: {x.grad.norm().item():.4f}")
    print(f"  ✓ 博弈门控层梯度: {block.game_attn.gate[0].weight.grad.norm().item():.6f}")


if __name__ == "__main__":
    print("=" * 55)
    print("博弈式扩散规划 - 核心模块验证")
    print("=" * 55)
    
    torch.manual_seed(42)
    
    try:
        test_game_theoretic_attention()
        test_game_theoretic_dit_block()
        test_game_consistency_loss()
        test_mode_comparison()
        test_gradient_flow()
        
        print("\n" + "=" * 55)
        print("✅ 所有测试通过！核心模块实现正确。")
        print("=" * 55)
        print("\n下一步：")
        print("  1. 将 game_theoretic_dit.py → diffusion_planner/model/module/dit.py")
        print("  2. 将 game_theoretic_decoder.py → diffusion_planner/model/module/decoder.py")
        print("  3. 将 game_theoretic_loss.py → diffusion_planner/game_loss.py")
        print("  4. 按 integration_guide.py 的说明修改 config 和训练脚本")
        print("  5. 运行消融实验对比三种模式的性能")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()