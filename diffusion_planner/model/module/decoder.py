"""
Game-Theoretic Decoder
======================
改造说明：
- 将原始 DiT 中的 DiTBlock 替换为 GameTheoreticDiTBlock
- 每个去噪 block 内都包含跨智能体博弈 attention
- 支持消融实验：通过 game_interaction_mode 控制博弈方式：
    - 'all': 每个 block 都做博弈（默认，即完整方法）  
    - 'last': 只有最后一个 block 做博弈（消融：博弈只发生一次）
    - 'none': 不做博弈（消融：退化为原始 Diffusion Planner）

用本文件替换 diffusion_planner/model/module/decoder.py
"""

import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath

from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.model.module.mixer import MixerBlock

# 导入改造后的模块
from diffusion_planner.model.module.dit import TimestepEmbedder, FinalLayer

# 导入新增的博弈模块
from diffusion_planner.model.module.dit import GameTheoreticDiTBlock, DiTBlock


class GameTheoreticDiT(nn.Module):
    """
    博弈式 DiT（Game-Theoretic Diffusion Transformer）。
    
    核心创新：将 Level-k 博弈推理嵌入到扩散去噪过程中。
    
    与原始 DiT 的差异：
    - blocks 使用 GameTheoreticDiTBlock 代替 DiTBlock
    - forward 中传入 neighbor_mask 用于博弈 attention 的 mask
    - 可选输出 game_attn_weights 用于可视化博弈交互
    
    Args:
        sde: 随机微分方程对象
        route_encoder: 路线编码器
        depth: DiT block 数量
        output_dim: 输出维度
        hidden_dim: 隐层维度
        heads: attention heads 数
        dropout: dropout 率
        model_type: 'x_start' 或 'score'
        game_interaction_mode: 博弈交互模式
            - 'all': 每个 block 都做博弈（完整方法）
            - 'last': 只有最后一个 block 做博弈（消融）
            - 'none': 不做博弈（退化为原始 Diffusion Planner，消融）
        use_role_embedding: 是否使用角色编码区分 ego/neighbor
    """
    def __init__(self, sde: SDE, route_encoder: nn.Module,
                 depth, output_dim, hidden_dim=192, heads=6, dropout=0.1,
                 mlp_ratio=4.0, model_type="x_start",
                 game_interaction_mode='all',
                 use_role_embedding=True):
        super().__init__()
        
        assert model_type in ["score", "x_start"]
        assert game_interaction_mode in ['all', 'last', 'none'], \
            f"game_interaction_mode must be 'all', 'last', or 'none', got {game_interaction_mode}"
        
        self._model_type = model_type
        self._game_interaction_mode = game_interaction_mode
        self.route_encoder = route_encoder
        
        # 与原始相同的组件
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512,
                           out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
        
        # 博弈式 DiT blocks
        # 根据 game_interaction_mode 决定每个 block 的类型
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if game_interaction_mode == 'none':
                # 消融：退化为原始 DiTBlock
                self.blocks.append(DiTBlock(hidden_dim, heads, dropout, mlp_ratio))
            elif game_interaction_mode == 'last':
                # 消融：只有最后一个 block 做博弈
                if i == depth - 1:
                    self.blocks.append(GameTheoreticDiTBlock(
                        hidden_dim, heads, dropout, mlp_ratio, use_role_embedding
                    ))
                else:
                    self.blocks.append(DiTBlock(hidden_dim, heads, dropout, mlp_ratio))
            else:  # 'all'：每个 block 都做博弈（完整方法）
                self.blocks.append(GameTheoreticDiTBlock(
                    hidden_dim, heads, dropout, mlp_ratio, use_role_embedding
                ))

    @property
    def model_type(self):
        return self._model_type

    @property
    def game_interaction_mode(self):
        return self._game_interaction_mode

    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask,
                return_game_weights=False):
        """
        Forward pass of GameTheoreticDiT.
        
        Args:
            x: (B, P, output_dim)  带噪轨迹
            t: (B,)  扩散时间步
            cross_c: (B, N, D)  场景条件（来自 encoder）
            route_lanes: 路线 lane 信息
            neighbor_current_mask: (B, P-1)  neighbor 是否存在的 mask（True=无效）
            return_game_weights: 是否返回博弈 attention 权重（用于可视化）
        
        Returns:
            x: (B, P, output_dim) 预测输出
            all_game_weights: list of (B, P, P)，每个 block 的博弈 attention 权重
        """
        B, P, _ = x.shape
        
        # === 与原始 DiT 相同的前处理 ===
        x = self.preproj(x)

        # 智能体位置编码（ego=0, neighbor=1）
        x_embedding = torch.cat([
            self.agent_embedding.weight[0][None, :],
            self.agent_embedding.weight[1][None, :].expand(P - 1, -1)
        ], dim=0)  # (P, D)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1)  # (B, P, D)
        x = x + x_embedding

        # 路线编码 + 时间步编码（作为 adaLN 条件）
        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding + self.t_embedder(t)  # (B, D)
        
        # 扩散时间步嵌入（单独传给博弈模块，用于调制博弈强度）
        diffusion_t_embed = self.t_embedder(t)  # (B, D)

        # self-attention mask：无效 neighbor 被屏蔽
        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask

        # neighbor mask for game attention（前置 False 给 ego，ego 永远有效）
        game_neighbor_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        game_neighbor_mask[:, 1:] = neighbor_current_mask

        # === 核心：博弈式去噪 ===
        all_game_weights = []
        
        for block in self.blocks:
            if isinstance(block, GameTheoreticDiTBlock):
                # 博弈式 block：每个智能体与其他智能体交互
                x, game_weights = block(
                    x, cross_c, y, attn_mask,
                    neighbor_mask=game_neighbor_mask,
                    diffusion_t_embed=diffusion_t_embed
                )
                if return_game_weights:
                    all_game_weights.append(game_weights)
            else:
                # 原始 block（消融实验用）
                x = block(x, cross_c, y, attn_mask)

        x = self.final_layer(x, y)

        if self._model_type == "score":
            x = x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)

        if return_game_weights:
            return x, all_game_weights
        return x


# ===================== RouteEncoder（与原始完全相同）=====================

class RouteEncoder(nn.Module):
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192,
                 tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()
        self._channel = channels_mlp_dim
        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim,
                                       out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len,
                                     hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim,
                                     act_layer=nn.GELU, drop=0.)
        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim,
                               out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        x = x[..., :4]
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)
        x = torch.mean(x, dim=1)
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x
        return x_result.view(B, -1)


# ===================== 改造后的 Decoder =====================

class Decoder(nn.Module):
    """
    博弈式 Diffusion Decoder。
    
    与原始 Decoder 的差异：
    - 使用 GameTheoreticDiT 代替原始 DiT
    - 支持通过 game_interaction_mode 控制博弈方式
    - 支持 return_game_weights 用于可视化分析
    
    新增 config 参数：
    - game_interaction_mode: 博弈交互模式（'all'/'last'/'none'）
    - use_role_embedding: 是否使用角色编码（默认 True）
    """
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()

        # 从 config 读取博弈参数（兼容旧 config，未设置时使用默认值）
        game_interaction_mode = getattr(config, 'game_interaction_mode', 'all')
        use_role_embedding = getattr(config, 'use_role_embedding', True)

        self.dit = GameTheoreticDiT(
            sde=self._sde,
            route_encoder=RouteEncoder(
                config.route_num, config.lane_len,
                drop_path_rate=config.encoder_drop_path_rate,
                hidden_dim=config.hidden_dim
            ),
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * 4,  # x, y, cos, sin
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type,
            game_interaction_mode=game_interaction_mode,
            use_role_embedding=use_role_embedding
        )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer
        self._guidance_fn = config.guidance_fn

        print(f"[GameTheoreticDecoder] game_interaction_mode='{game_interaction_mode}', "
              f"use_role_embedding={use_role_embedding}")

    @property
    def sde(self):
        return self._sde

    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process（与原始接口完全兼容）。
        
        新增功能：
        - 训练时加入博弈一致性分析（通过 return_game_weights）
        - 推理时支持可视化博弈 attention（设置 inputs['return_game_weights']=True）
        """
        # 提取当前状态（与原始完全相同）
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1)  # [B, P, 4]
        B, P, _ = current_states.shape

        # 提取 encoder 输出和路线信息
        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        if self.training:
            # ===== 训练阶段 =====
            sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1)
            diffusion_time = inputs['diffusion_time']

            score = self.dit(
                sampled_trajectories,
                diffusion_time,
                ego_neighbor_encoding,
                route_lanes,
                neighbor_current_mask
            )

            return {
                "score": score.reshape(B, P, -1, 4)
            }

        else:
            # ===== 推理阶段 =====
            # 初始化噪声轨迹（与原始相同）
            xT = torch.cat([
                current_states[:, :, None],
                torch.randn(B, P, self._future_len, 4).to(current_states.device) * 0.5
            ], dim=2).reshape(B, P, -1)

            # 当前状态约束（ego 和 neighbor 的起始位置固定）
            def initial_state_constraint(xt, t, step):
                xt = xt.reshape(B, P, -1, 4)
                xt[:, :, 0, :] = current_states
                return xt.reshape(B, P, -1)

            # DPM 采样（与原始相同的采样流程，博弈已嵌入 DiT forward 中）
            x0 = dpm_sampler(
                self.dit,
                xT,
                other_model_params={
                    "cross_c": ego_neighbor_encoding,
                    "route_lanes": route_lanes,
                    "neighbor_current_mask": neighbor_current_mask
                },
                dpm_solver_params={
                    "correcting_xt_fn": initial_state_constraint,
                },
                model_wrapper_params={
                    "classifier_fn": self._guidance_fn,
                    "classifier_kwargs": {
                        "model": self.dit,
                        "model_condition": {
                            "cross_c": ego_neighbor_encoding,
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask
                        },
                        "inputs": inputs,
                        "observation_normalizer": self._observation_normalizer,
                        "state_normalizer": self._state_normalizer
                    },
                    "guidance_scale": 0.5,
                    "guidance_type": "classifier" if self._guidance_fn is not None else "uncond"
                },
            )
            x0 = self._state_normalizer.inverse(x0.reshape(B, P, -1, 4))[:, :, 1:]

            return {
                "prediction": x0
            }
        

class DiT(nn.Module):
    def __init__(self, sde: SDE, route_encoder: nn.Module, depth, output_dim, hidden_dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, model_type="x_start"):
        super().__init__()
        
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
               
    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask):
        """
        Forward pass of DiT.
        x: (B, P, output_dim)   -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        """
        B, P, _ = x.shape
        
        x = self.preproj(x)

        x_embedding = torch.cat([self.agent_embedding.weight[0][None, :], self.agent_embedding.weight[1][None, :].expand(P - 1, -1)], dim=0)  # (P, D)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1) # (B, P, D)
        x = x + x_embedding     

        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding
        y = y + self.t_embedder(t)      

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask
        
        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)  
            
        x = self.final_layer(x, y)
        
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
