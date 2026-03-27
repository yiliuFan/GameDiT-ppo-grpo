from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


EPS = 1e-6


def _scale_from_unit_interval(samples: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * samples


class ExplorationPolicy(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        input_dim = hidden_dim * 2
        self.feature_norm = nn.LayerNorm(input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(input_dim, 6)
        self.value_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
        )
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def _pool_scene(self, scene_encoding: torch.Tensor) -> torch.Tensor:
        scene_mean = scene_encoding.mean(dim=1)
        scene_max = scene_encoding.amax(dim=1)
        return torch.cat([scene_mean, scene_max], dim=-1)

    def forward(self, scene_encoding: torch.Tensor) -> dict:
        scene_feature = self._pool_scene(scene_encoding)
        hidden = self.backbone(self.feature_norm(scene_feature))
        logits = self.policy_head(hidden)
        alpha_beta = F.softplus(logits[:, :4]) + 1.0

        return {
            "scene_feature": scene_feature,
            "policy_feature": hidden,
            "lateral_alpha": alpha_beta[:, 0],
            "lateral_beta": alpha_beta[:, 1],
            "longitudinal_alpha": alpha_beta[:, 2],
            "longitudinal_beta": alpha_beta[:, 3],
            "guidance_strength": 0.25 + 0.75 * torch.sigmoid(logits[:, 4]),
            "noise_scale": 0.50 + 1.00 * torch.sigmoid(logits[:, 5]),
            "state_value": self.value_head(hidden).squeeze(-1),
        }


def _get_beta_distributions(policy_output: dict) -> tuple:
    return (
        Beta(policy_output["lateral_alpha"], policy_output["lateral_beta"]),
        Beta(policy_output["longitudinal_alpha"], policy_output["longitudinal_beta"]),
    )


def log_prob_of_actions(policy_output: dict, lateral_unit: torch.Tensor, longitudinal_unit: torch.Tensor) -> tuple:
    lateral_dist, longitudinal_dist = _get_beta_distributions(policy_output)
    lateral_unit = lateral_unit.clamp(EPS, 1.0 - EPS)
    longitudinal_unit = longitudinal_unit.clamp(EPS, 1.0 - EPS)

    log_prob = lateral_dist.log_prob(lateral_unit) + longitudinal_dist.log_prob(longitudinal_unit)
    entropy = lateral_dist.entropy() + longitudinal_dist.entropy()
    return log_prob, entropy


def sample_policy(
    policy_output: dict,
    lateral_min: float,
    lateral_max: float,
    longitudinal_min: float,
    longitudinal_max: float,
) -> dict:
    lateral_dist, longitudinal_dist = _get_beta_distributions(policy_output)

    lateral_unit = lateral_dist.sample().clamp(EPS, 1.0 - EPS)
    longitudinal_unit = longitudinal_dist.sample().clamp(EPS, 1.0 - EPS)
    log_prob, entropy = log_prob_of_actions(policy_output, lateral_unit, longitudinal_unit)

    return {
        **policy_output,
        "lateral_unit": lateral_unit,
        "longitudinal_unit": longitudinal_unit,
        "lateral_scale": _scale_from_unit_interval(lateral_unit, lateral_min, lateral_max),
        "longitudinal_scale": _scale_from_unit_interval(longitudinal_unit, longitudinal_min, longitudinal_max),
        "log_prob": log_prob,
        "entropy": entropy,
    }
