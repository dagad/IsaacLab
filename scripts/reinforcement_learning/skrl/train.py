# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Multi-GPU Distributed RL Training with skrl.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import random
import time
import torch
import torch.distributed as dist
import torch.nn.parallel

from isaaclab.app import AppLauncher

# ✅ DDP 초기화 함수
def setup_distributed():
    """PyTorch DDP (Distributed Data Parallel) 초기화"""
    if torch.cuda.is_available() and torch.distributed.is_available():
        dist.init_process_group(backend="nccl")  # GPU 간 통신 최적화
        local_rank = int(os.environ["LOCAL_RANK"])  # torchrun이 제공하는 환경 변수
        torch.cuda.set_device(local_rank)  # 각 프로세스를 해당 GPU에 매핑
        print(f"🔥 Process {dist.get_rank()} initialized on GPU {local_rank}")

# ✅ DDP 종료 함수
def cleanup_distributed():
    dist.destroy_process_group()

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--distributed", action="store_true", default=False, help="Enable Multi-GPU training.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--max_iterations", type=int, default=1000, help="RL training iterations.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# DDP 초기화 (Multi-GPU 모드일 경우)
if args_cli.distributed:
    setup_distributed()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from datetime import datetime
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. Install 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device

    # Multi-GPU 설정
    local_rank = 0
    if args_cli.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        env_cfg.sim.device = f"cuda:{local_rank}"

    # Max iterations 설정
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # Random seed 설정
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # Logging setup (GPU 0에서만 출력)
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir

    # Logging files (GPU 0에서만 저장)
    if local_rank == 0:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Get checkpoint path
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # Create RL environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm == "ppo":
        env = multi_agent_to_single_agent(env)

    # Wrap environment
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # Multi-GPU 지원을 위한 DistributedDataParallel (DDP) 적용
    model = Runner(env, agent_cfg).agent
    if args_cli.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Load checkpoint
    if resume_path and local_rank == 0:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        model.load(resume_path)

    # Train
    start_time = time.time()
    model.learn()
    elapsed_time = time.time() - start_time

    # GPU 0에서만 학습 시간 기록
    if local_rank == 0:
        with open(os.path.join(log_dir, "training_time.log"), "w") as f:
            f.write(f"🔥 Total training time: {elapsed_time:.2f} seconds\n")
        print(f"🔥 Total training time: {elapsed_time:.2f} seconds")

    # DDP 정리
    if args_cli.distributed:
        cleanup_distributed()

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
