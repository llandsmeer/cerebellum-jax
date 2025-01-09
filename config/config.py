from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from pathlib import Path
import tomllib


@dataclass
class ModelConfig:
    name: str = "JAXCartpoleModel"  # Model class name
    params: Dict[str, Any] = field(default_factory=dict)  # Arbitrary model parameters


@dataclass
class TrainerConfig:
    gamma: float = 0.99
    max_episodes: int = 100
    max_steps_per_episode: int = 500


@dataclass
class CallbackConfig:
    checkpoint_freq: int = 10
    checkpoint_path: str = "checkpoints"
    log_file: str = "training_log.csv"


@dataclass
class VisualizationConfig:
    enabled: bool = False
    num_episodes: int = 3
    render_mode: str = "human"


@dataclass
class EvalConfig:
    checkpoint_path: Optional[str] = None
    num_episodes: int = 10
    render: bool = True


@dataclass
class Config:
    # Mode configuration
    mode: str = "train"  # "train" or "eval"
    config_file: Optional[str] = None

    # Environment configuration
    env_id: str = "CartPole-v1"

    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_toml(cls, path: Union[str, Path]):
        with open(path, "rb") as f:  # Open in binary mode for tomllib
            config_dict = tomllib.load(f)

        # Convert nested dicts to dataclass instances
        model_dict = config_dict.pop("model", {})
        model_config = ModelConfig(
            name=model_dict.pop("name", "JAXCartpoleModel"),
            params=model_dict,  # All remaining parameters go into params dict
        )

        trainer_config = TrainerConfig(**config_dict.pop("trainer", {}))
        callback_config = CallbackConfig(**config_dict.pop("callbacks", {}))
        viz_config = VisualizationConfig(**config_dict.pop("visualization", {}))
        eval_config = EvalConfig(**config_dict.pop("eval", {}))

        return cls(
            model=model_config,
            trainer=trainer_config,
            callbacks=callback_config,
            visualization=viz_config,
            eval=eval_config,
            **config_dict,
        )
