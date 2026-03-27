from __future__ import annotations

import io
import os
from typing import Any, Dict, Tuple

import torch
from mmengine import fileio


def _resolve_path(path: str) -> str:
    return os.path.join(path, "latest.pth") if os.path.isdir(path) else path


def _adapt_state_dict(module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    module_keys = list(module.state_dict().keys())
    if not module_keys or not state_dict:
        return state_dict

    module_has_prefix = module_keys[0].startswith("module.")
    state_keys = list(state_dict.keys())
    state_has_prefix = state_keys[0].startswith("module.")

    if module_has_prefix == state_has_prefix:
        return state_dict
    if module_has_prefix and not state_has_prefix:
        return {f"module.{key}": value for key, value in state_dict.items()}
    return {key[len("module."):]: value for key, value in state_dict.items() if key.startswith("module.")}


def load_planner_checkpoint(model, path: str, device: str, prefer_ema: bool = True) -> Dict[str, Any]:
    checkpoint_path = _resolve_path(path)
    checkpoint_bytes = fileio.get(checkpoint_path)
    with io.BytesIO(checkpoint_bytes) as buffer:
        checkpoint = torch.load(buffer, map_location=device)

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if prefer_ema and checkpoint.get("ema_state_dict") is not None:
            state_dict = checkpoint["ema_state_dict"]
        elif checkpoint.get("model") is not None:
            state_dict = checkpoint["model"]

    model.load_state_dict(_adapt_state_dict(model, state_dict), strict=False)
    return checkpoint if isinstance(checkpoint, dict) else {}


def save_rl_checkpoint(
    model,
    exploration_policy,
    optimizer,
    policy_optimizer,
    scheduler,
    save_path: str,
    epoch: int,
    metrics: Dict[str, Any],
    wandb_id: str,
    ema=None,
    reference_model=None,
) -> None:
    payload = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "exploration_policy": exploration_policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "policy_optimizer": policy_optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "wandb_id": wandb_id,
    }
    if ema is not None:
        payload["ema_state_dict"] = ema.ema.state_dict()
    if reference_model is not None:
        payload["reference_model"] = reference_model.state_dict()

    with io.BytesIO() as buffer:
        torch.save(payload, buffer)
        bytes_value = buffer.getvalue()
        metric_name = metrics.get("reward_mean", metrics.get("loss", 0.0))
        if isinstance(metric_name, torch.Tensor):
            metric_name = metric_name.item()
        fileio.put(bytes_value, f"{save_path}/model_epoch_{epoch + 1}_metric_{metric_name:.4f}.pth")
        fileio.put(bytes_value, f"{save_path}/latest.pth")


def resume_rl_checkpoint(path: str, model, exploration_policy, optimizer, policy_optimizer, scheduler, ema, reference_model, device: str) -> Tuple[int, str, bool]:
    checkpoint_path = _resolve_path(path)
    checkpoint_bytes = fileio.get(checkpoint_path)
    with io.BytesIO(checkpoint_bytes) as buffer:
        checkpoint = torch.load(buffer, map_location=device)

    if checkpoint.get("model") is not None:
        model.load_state_dict(_adapt_state_dict(model, checkpoint["model"]), strict=False)
    if checkpoint.get("exploration_policy") is not None:
        exploration_policy.load_state_dict(_adapt_state_dict(exploration_policy, checkpoint["exploration_policy"]), strict=False)
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("policy_optimizer") is not None:
        policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if ema is not None and checkpoint.get("ema_state_dict") is not None:
        ema.ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.ema.eval()
        for parameter in ema.ema.parameters():
            parameter.requires_grad_(False)

    loaded_reference = False
    if reference_model is not None and checkpoint.get("reference_model") is not None:
        reference_model.load_state_dict(_adapt_state_dict(reference_model, checkpoint["reference_model"]), strict=False)
        loaded_reference = True

    return checkpoint.get("epoch", 0), checkpoint.get("wandb_id"), loaded_reference
