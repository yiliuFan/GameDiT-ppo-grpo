"""
Config 补丁 & 集成使用说明
===========================
将本文件内容添加到你的 config 类中，并按照说明修改相关训练脚本。
"""

# ===========================================================
# 1. 修改 diffusion_planner/utils/config.py
# 在 Config 类中新增以下参数
# ===========================================================

ADDITIONAL_CONFIG_PARAMS = """
# 在 Config 类的 __init__ 或 from_args() 中添加：

# 博弈交互模式（核心创新参数）
# 'all'  - 每个 DiT block 都做博弈 attention（完整方法，推荐）
# 'last' - 只有最后一个 block 做博弈（消融实验：博弈只发生一次）
# 'none' - 不做博弈（消融实验：退化为原始 Diffusion Planner）
self.game_interaction_mode = 'all'

# 是否区分 ego 和 neighbor 的角色编码
self.use_role_embedding = True

# 博弈一致性损失权重（在总损失中的系数）
self.game_loss_weight = 0.1
"""

# ===========================================================
# 2. 修改 train_predictor.py 的 argparse 部分
# ===========================================================

ADDITIONAL_ARGPARSE_PARAMS = """
# 在 get_args() 中添加：
parser.add_argument('--game_interaction_mode', type=str, 
                    choices=['all', 'last', 'none'],
                    default='all',
                    help='博弈交互模式: all=每个block, last=最后block, none=无博弈(消融)')
parser.add_argument('--game_loss_weight', type=float, default=0.1,
                    help='博弈一致性损失权重 lambda（默认 0.1）')
parser.add_argument('--use_role_embedding', type=boolean, default=True,
                    help='是否使用 ego/neighbor 角色编码')
"""

# ===========================================================
# 3. 修改 diffusion_planner/train_epoch.py
# ===========================================================

TRAIN_EPOCH_MODIFICATION = """
# 在 train_epoch.py 中，找到计算 total_loss 的地方，改为：

# 原始代码（大概是）:
# total_loss = alpha_planning * ego_loss + neighbor_loss

# 改为：
alpha_planning = ...  # 原有参数
ego_loss = loss.get("ego_planning_loss", 0)
neighbor_loss = loss.get("neighbor_prediction_loss", 0)
game_loss = loss.get("game_consistency_loss", 0)
game_weight = loss.get("game_loss_weight", 0.1)

total_loss = (alpha_planning * ego_loss 
              + neighbor_loss 
              + game_weight * game_loss)
"""

# ===========================================================
# 4. 文件替换说明
# ===========================================================

FILE_REPLACEMENT_GUIDE = """
将以下文件复制到对应位置：

1. game_theoretic_dit.py 
   → diffusion_planner/model/module/dit.py
   （替换原始 dit.py，完全向下兼容）

2. game_theoretic_decoder.py
   → diffusion_planner/model/module/decoder.py
   （替换原始 decoder.py，需确保 Config 中有新增参数）

3. game_theoretic_loss.py
   → diffusion_planner/game_loss.py
   （新文件，在 train_epoch.py 中 import 使用）
"""

# ===========================================================
# 5. 消融实验配置（直接复制对应命令运行）
# ===========================================================

ABLATION_CONFIGS = {
    "完整方法（论文主要结果）": {
        "game_interaction_mode": "all",
        "use_role_embedding": True,
        "game_loss_weight": 0.1,
    },
    "消融1：退化为原始Diffusion Planner": {
        "game_interaction_mode": "none",
        "use_role_embedding": False,
        "game_loss_weight": 0.0,
    },
    "消融2：博弈只在最后一步发生": {
        "game_interaction_mode": "last",
        "use_role_embedding": True,
        "game_loss_weight": 0.1,
    },
    "消融3：无角色编码": {
        "game_interaction_mode": "all",
        "use_role_embedding": False,
        "game_loss_weight": 0.1,
    },
    "消融4：无博弈一致性损失": {
        "game_interaction_mode": "all",
        "use_role_embedding": True,
        "game_loss_weight": 0.0,
    },
}

ABLATION_COMMANDS = """
# 消融实验运行命令（在 torch_run.sh 基础上修改参数）：

# 完整方法
python train_predictor.py --game_interaction_mode all --game_loss_weight 0.1 --name game_all

# 消融1：退化为原始
python train_predictor.py --game_interaction_mode none --game_loss_weight 0.0 --name ablation_none

# 消融2：博弈只在最后
python train_predictor.py --game_interaction_mode last --game_loss_weight 0.1 --name ablation_last

# 消融3：无角色编码
python train_predictor.py --game_interaction_mode all --use_role_embedding False --name ablation_norole

# 消融4：无博弈损失
python train_predictor.py --game_interaction_mode all --game_loss_weight 0.0 --name ablation_noloss
"""

if __name__ == "__main__":
    print("=" * 60)
    print("博弈式扩散规划集成指南")
    print("=" * 60)
    print("\n【文件替换】")
    print(FILE_REPLACEMENT_GUIDE)
    print("\n【Config 新增参数】")
    print(ADDITIONAL_CONFIG_PARAMS)
    print("\n【ArgParse 新增参数】")
    print(ADDITIONAL_ARGPARSE_PARAMS)
    print("\n【train_epoch.py 修改】")
    print(TRAIN_EPOCH_MODIFICATION)
    print("\n【消融实验命令】")
    print(ABLATION_COMMANDS)