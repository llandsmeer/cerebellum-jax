# main.py

import gymnasium as gym
import argparse
from pathlib import Path
from typing import Optional, Type

from trainer.trainer import Trainer
from trainer.callbacks import CheckpointCallback, LoggingCallback
from models.jax_basemodel import JAXCartpoleModel
from models.base_model import BaseModel
from config.config import (
    Config,
    ModelConfig,
    TrainerConfig,
    CallbackConfig,
    VisualizationConfig,
)


def get_model_class(model_name: str) -> Type[BaseModel]:
    """Get model class by name."""
    model_map = {
        "JAXCartpoleModel": JAXCartpoleModel,
        # Add other models here
    }
    if model_name not in model_map:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(model_map.keys())}"
        )
    return model_map[model_name]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="RL training script")

    # Mode configuration
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Running mode: train or eval",
    )
    parser.add_argument("--config", type=str, help="Path to TOML config file")

    # Environment
    parser.add_argument(
        "--env-id", type=str, default="CartPole-v1", help="Gymnasium environment ID"
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="JAXCartpoleModel",
        help="Name of the model class to use",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        nargs="+",
        default=[],
        help="Model parameters in key=value format (e.g., lr=0.001 hidden_sizes=64,32)",
    )

    # Training configuration
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Hidden layer sizes",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training configuration
    parser.add_argument(
        "--max-episodes", type=int, default=100, help="Maximum number of episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Maximum steps per episode"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Callback configuration
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Checkpoint frequency in episodes",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints",
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--log-file", type=str, default="training_log.csv", help="Path to log file"
    )

    # Visualization configuration
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument(
        "--viz-episodes", type=int, default=3, help="Number of episodes to visualize"
    )

    args = parser.parse_args()

    # If config file is provided, load it and override with CLI args
    if args.config:
        config = Config.from_toml(args.config)
    else:
        # Parse model parameters from command line
        model_params = {}
        for param in args.model_params:
            key, value = param.split("=")
            # Try to convert to appropriate type
            try:
                value = eval(value)
            except:
                pass  # Keep as string if conversion fails
            model_params[key] = value

        # Create config from CLI args
        config = Config(
            mode=args.mode,
            env_id=args.env_id,
            model=ModelConfig(name=args.model_name, params=model_params),
            trainer=TrainerConfig(
                gamma=args.gamma,
                max_episodes=args.max_episodes,
                max_steps_per_episode=args.max_steps,
            ),
            callbacks=CallbackConfig(
                checkpoint_freq=args.checkpoint_freq,
                checkpoint_path=args.checkpoint_path,
                log_file=args.log_file,
            ),
            visualization=VisualizationConfig(
                enabled=args.visualize, num_episodes=args.viz_episodes
            ),
        )

    return config


def main():
    config = parse_args()

    # Create environment
    env = gym.make(
        config.env_id, render_mode="human" if config.visualization.enabled else None
    )

    # Create model
    model_class = get_model_class(config.model.name)
    model = model_class(**config.model.params)  # Pass all parameters directly

    # Set up callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=config.callbacks.checkpoint_freq,
            save_path=str(
                Path(config.callbacks.checkpoint_path) / "model_checkpoint.pkl"
            ),
        ),
        LoggingCallback(log_file=config.callbacks.log_file),
    ]

    # Create trainer
    trainer = Trainer(
        model=model,
        env=env,
        callbacks=callbacks,
        gamma=config.trainer.gamma,
        max_episodes=config.trainer.max_episodes,
        max_steps_per_episode=config.trainer.max_steps_per_episode,
    )

    if config.mode == "train":
        trainer.train()

        if config.visualization.enabled:
            trainer.visualize(num_episodes=config.visualization.num_episodes)
    else:  # eval mode
        trainer.visualize(num_episodes=config.visualization.num_episodes)


if __name__ == "__main__":
    main()
