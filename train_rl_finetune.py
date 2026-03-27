import os
import argparse

import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from timm.utils import ModelEma

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.rl.exploration_policy import ExplorationPolicy
from diffusion_planner.rl.train_epoch_rl import train_epoch_rl
from diffusion_planner.rl.checkpoint import load_planner_checkpoint, resume_rl_checkpoint, save_rl_checkpoint
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger
from diffusion_planner.utils.train_utils import set_seed


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def set_module_trainable(module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def get_args():
    parser = argparse.ArgumentParser(description="PlannerRFT-style RL fine-tuning for Diffusion Planner")

    parser.add_argument("--game_interaction_mode", type=str, choices=["all", "last", "none"], default="all")
    parser.add_argument("--game_loss_weight", type=float, default=0.1)
    parser.add_argument("--use_role_embedding", type=boolean, default=True)

    parser.add_argument("--name", type=str, default="diffusion-planner-rft")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--resume_model_path", type=str, default=None)

    parser.add_argument("--train_set", type=str, default=None)
    parser.add_argument("--train_set_list", type=str, default=None)
    parser.add_argument("--future_len", type=int, default=80)
    parser.add_argument("--time_len", type=int, default=21)

    parser.add_argument("--agent_state_dim", type=int, default=11)
    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--static_objects_state_dim", type=int, default=10)
    parser.add_argument("--static_objects_num", type=int, default=5)
    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_state_dim", type=int, default=12)
    parser.add_argument("--lane_num", type=int, default=70)
    parser.add_argument("--route_len", type=int, default=20)
    parser.add_argument("--route_state_dim", type=int, default=12)
    parser.add_argument("--route_num", type=int, default=25)

    parser.add_argument("--augment_prob", type=float, default=0.5)
    parser.add_argument("--normalization_file_path", type=str, default="normalization.json")
    parser.add_argument("--use_data_augment", type=boolean, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--save_utd", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--rl_policy_learning_rate", type=float, default=1e-4)
    parser.add_argument("--warm_up_epoch", type=int, default=3)
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--alpha_planning_loss", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", type=boolean, default=True)

    parser.add_argument("--encoder_depth", type=int, default=3)
    parser.add_argument("--decoder_depth", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--diffusion_model_type", type=str, choices=["score", "x_start"], default="x_start")
    parser.add_argument("--predicted_neighbor_num", type=int, default=10)

    parser.add_argument("--use_wandb", type=boolean, default=False)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--ddp", type=boolean, default=True)
    parser.add_argument("--port", type=str, default="22323")

    parser.add_argument("--rl_num_samples", type=int, default=4)
    parser.add_argument("--rl_ddim_steps", type=int, default=5)
    parser.add_argument("--rl_reference_mode", type=str, choices=["expert", "model"], default="expert")
    parser.add_argument("--rl_bc_weight", type=float, default=1.0)
    parser.add_argument("--rl_trajectory_weight", type=float, default=0.5)
    parser.add_argument("--rl_policy_weight", type=float, default=0.2)
    parser.add_argument("--rl_entropy_weight", type=float, default=1e-3)
    parser.add_argument("--rl_normalize_advantage", type=boolean, default=True)
    parser.add_argument("--rl_reward_clip", type=float, default=10.0)
    parser.add_argument("--rl_sampling_eta", type=float, default=0.15)
    parser.add_argument("--rl_init_noise_scale", type=float, default=0.5)
    parser.add_argument("--rl_guidance_power", type=float, default=1.5)
    parser.add_argument("--rl_diffusion_eps", type=float, default=1e-3)
    parser.add_argument("--rl_grad_clip", type=float, default=5.0)
    parser.add_argument("--rl_lateral_scale_min", type=float, default=0.05)
    parser.add_argument("--rl_lateral_scale_max", type=float, default=1.25)
    parser.add_argument("--rl_longitudinal_scale_min", type=float, default=0.05)
    parser.add_argument("--rl_longitudinal_scale_max", type=float, default=1.00)
    parser.add_argument("--rl_collision_distance", type=float, default=4.0)
    parser.add_argument("--rl_reward_progress_weight", type=float, default=0.05)
    parser.add_argument("--rl_reward_route_weight", type=float, default=0.50)
    parser.add_argument("--rl_reward_collision_weight", type=float, default=2.00)
    parser.add_argument("--rl_reward_imitation_weight", type=float, default=0.25)
    parser.add_argument("--rl_reward_smoothness_weight", type=float, default=0.05)
    parser.add_argument("--rl_reward_heading_weight", type=float, default=0.10)
    parser.add_argument("--rl_ppo_epochs", type=int, default=4)
    parser.add_argument("--rl_ppo_clip", type=float, default=0.2)
    parser.add_argument("--rl_value_weight", type=float, default=0.5)
    parser.add_argument("--rl_value_clip", type=float, default=0.2)
    parser.add_argument("--rl_grpo_epochs", type=int, default=2)
    parser.add_argument("--rl_grpo_clip", type=float, default=0.2)
    parser.add_argument("--rl_reference_kl_weight", type=float, default=0.01)
    parser.add_argument("--rl_finetune_decoder_only", type=boolean, default=True)

    args = parser.parse_args()
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    args.guidance_fn = None
    return args


def model_training(args):
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    args.runtime_device = f"cuda:{rank}" if args.device == "cuda" else args.device

    if global_rank == 0:
        print(f"------------- {args.name} -------------")
        print(f"Batch size: {args.batch_size}")
        print(f"Planner LR: {args.learning_rate}")
        print(f"Policy LR: {args.rl_policy_learning_rate}")
        print(f"RL samples: {args.rl_num_samples}")
        print(f"Reference mode: {args.rl_reference_mode}")
        print(f"PPO epochs: {args.rl_ppo_epochs}")
        print(f"GRPO epochs: {args.rl_grpo_epochs}")
        print(f"Runtime device: {args.runtime_device}")

        if args.resume_model_path is not None:
            save_path = args.resume_model_path
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{args.save_dir}/training_log/{args.name}/{timestamp}/"
            os.makedirs(save_path, exist_ok=True)

        args_dict = vars(args)
        args_dict = {
            key: value if not isinstance(value, (StateNormalizer, ObservationNormalizer)) else value.to_dict()
            for key, value in args_dict.items()
        }
        from mmengine.fileio import dump

        dump(args_dict, os.path.join(save_path, "args.json"), file_format="json", indent=4)
    else:
        save_path = None

    set_seed(args.seed + global_rank)

    aug = StatePerturbation(augment_prob=args.augment_prob, device=args.runtime_device) if args.use_data_augment else None
    train_set = DiffusionPlannerData(
        args.train_set,
        args.train_set_list,
        args.agent_num,
        args.predicted_neighbor_num,
        args.future_len,
    )
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=ddp.get_world_size(),
        rank=global_rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=args.batch_size // ddp.get_world_size(),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if global_rank == 0:
        print(f"Dataset Prepared: {len(train_set)} train data\n")

    if args.ddp:
        torch.distributed.barrier()

    diffusion_planner = Diffusion_Planner(args)
    reference_planner = Diffusion_Planner(args)
    old_planner = Diffusion_Planner(args)
    exploration_policy = ExplorationPolicy(args.hidden_dim)
    reference_loaded = False

    if args.pretrained_model_path is not None:
        if args.resume_model_path is None:
            load_planner_checkpoint(diffusion_planner, args.pretrained_model_path, args.runtime_device, prefer_ema=args.use_ema)
        load_planner_checkpoint(reference_planner, args.pretrained_model_path, args.runtime_device, prefer_ema=args.use_ema)
        load_planner_checkpoint(old_planner, args.pretrained_model_path, args.runtime_device, prefer_ema=args.use_ema)
        reference_loaded = True
        if global_rank == 0 and args.resume_model_path is None:
            print(f"Loaded pretrained planner from {args.pretrained_model_path}")

    if args.resume_model_path is not None:
        init_epoch = 0
        wandb_id = None
    else:
        init_epoch = 0
        wandb_id = None

    if not reference_loaded:
        reference_planner.load_state_dict(diffusion_planner.state_dict(), strict=False)
        old_planner.load_state_dict(diffusion_planner.state_dict(), strict=False)

    if args.rl_finetune_decoder_only:
        set_module_trainable(diffusion_planner.encoder, False)

    set_module_trainable(reference_planner, False)
    set_module_trainable(old_planner, False)
    reference_planner.eval()
    old_planner.eval()

    diffusion_planner = diffusion_planner.to(args.runtime_device)
    reference_planner = reference_planner.to(args.runtime_device)
    old_planner = old_planner.to(args.runtime_device)
    exploration_policy = exploration_policy.to(args.runtime_device)

    if args.ddp:
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank], find_unused_parameters=True)
        exploration_policy = DDP(exploration_policy, device_ids=[rank], find_unused_parameters=True)

    model_ema = None
    if args.use_ema:
        model_ema = ModelEma(diffusion_planner, decay=0.999, device=args.runtime_device)

    planner_parameters = [parameter for parameter in ddp.get_model(diffusion_planner, args.ddp).parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(planner_parameters, lr=args.learning_rate)
    policy_optimizer = optim.AdamW(ddp.get_model(exploration_policy, args.ddp).parameters(), lr=args.rl_policy_learning_rate)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, args.train_epochs, args.warm_up_epoch)

    if args.resume_model_path is not None:
        init_epoch, wandb_id, loaded_reference_from_ckpt = resume_rl_checkpoint(
            args.resume_model_path,
            diffusion_planner,
            exploration_policy,
            optimizer,
            policy_optimizer,
            scheduler,
            model_ema,
            reference_planner,
            args.runtime_device,
        )
        if loaded_reference_from_ckpt:
            reference_loaded = True
        if not reference_loaded and args.pretrained_model_path is None:
            reference_planner.load_state_dict(ddp.get_model(diffusion_planner, args.ddp).state_dict(), strict=False)
        old_planner.load_state_dict(ddp.get_model(diffusion_planner, args.ddp).state_dict(), strict=False)
        if global_rank == 0:
            print(f"Resumed RL checkpoint from {args.resume_model_path}")

    if global_rank == 0:
        planner_params = sum(p.numel() for p in ddp.get_model(diffusion_planner, args.ddp).parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in ddp.get_model(diffusion_planner, args.ddp).parameters() if not p.requires_grad)
        policy_params = sum(p.numel() for p in ddp.get_model(exploration_policy, args.ddp).parameters())
        print(f"Trainable Planner Params: {planner_params}")
        print(f"Frozen Planner Params: {frozen_params}")
        print(f"Policy Params: {policy_params}")

    wandb_logger = Logger(args.name, args.notes, args, wandb_resume_id=wandb_id, save_path=save_path, rank=global_rank)

    if args.ddp:
        torch.distributed.barrier()

    for epoch in range(init_epoch, args.train_epochs):
        if global_rank == 0:
            print(f"Epoch {epoch + 1}/{args.train_epochs}")

        train_loss, _ = train_epoch_rl(
            train_loader,
            diffusion_planner,
            exploration_policy,
            optimizer,
            policy_optimizer,
            args,
            model_ema,
            aug,
            reference_model=reference_planner,
            old_model=old_planner,
        )

        if global_rank == 0:
            lr_dict = {
                "planner_lr": optimizer.param_groups[0]["lr"],
                "policy_lr": policy_optimizer.param_groups[0]["lr"],
            }
            wandb_logger.log_metrics({f"train_rl/{k}": v for k, v in train_loss.items()}, step=epoch + 1)
            wandb_logger.log_metrics({f"lr/{k}": v for k, v in lr_dict.items()}, step=epoch + 1)

            if (epoch + 1) % args.save_utd == 0:
                save_rl_checkpoint(
                    diffusion_planner,
                    exploration_policy,
                    optimizer,
                    policy_optimizer,
                    scheduler,
                    save_path,
                    epoch,
                    train_loss,
                    wandb_logger.id,
                    model_ema,
                    reference_model=reference_planner,
                )
                print(f"Model saved in {save_path}\n")

        scheduler.step()
        train_sampler.set_epoch(epoch + 1)

    if global_rank == 0:
        wandb_logger.finish()


if __name__ == "__main__":
    model_training(get_args())
