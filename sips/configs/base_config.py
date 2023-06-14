import os
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# ARCH


@dataclass
class _ArchConfig:
    seed: int = 42  #                        # Random seed
    max_epochs: int = 50  #                  # Maximum number of epochs


# --------------------------------------------------------------------------
# WANDB


@dataclass
class _WandBConfig:
    name: str = ""  #                              # WandB run name
    dry_run: bool = True  #                        # WandB dry-run (no logging)
    project_os_env: str = "WANDB_PROJECT"  #       # WandB project (from os.env)
    entity_os_env: str = "WANDB_ENTITY"  #         # WandB entity (from os.env)
    tags: list[str] = field(default_factory=list)  # WandB tags
    save_dir: str = ""  #                          # WandB save folder

    @property
    def project(self) -> str:
        return os.environ.get(self.project_os_env, "")

    @property
    def entity(self) -> str:
        return os.environ.get(self.entity_os_env, "")


# --------------------------------------------------------------------------
# MODEL


@dataclass
class _ModelConfig:
    # Checkpointing
    checkpoint_path: str = "/data/experiments/sips/"
    save_checkpoint: bool = True

    # Model parameters
    keypoint_loss_weight: float = 2.0  #     # Keypoint loss weight
    descriptor_loss_weight: float = 1.0  #   # Descriptor loss weight
    score_loss_weight = 1.0  #               # Score loss weight
    use_color: bool = False  #               # Use color or grayscale images
    with_io: bool = True  #                  # Use IONet
    do_upsample: bool = True  #              # Upsample descriptors
    do_cross: bool = True  #                 # Use cross-border keypoints
    descriptor_loss: bool = True  #          # Use hardest neg. mining descriptor loss
    keypoint_net_type: str = "KeypointNet"  ## Type of keypoint network. Supported ['KeypointNet', 'KeypointResnet']

    # Optimizer
    opt_learn_rate: float = 0.001
    opt_weight_decay: float = 0.0

    # Scheduler
    sched_decay_rate: float = 0.5  #         # Scheduler decay rate
    sched_decay_frequency: int = 40  #       # Number of epochs when to decay the initial learning rate by decay rate


# --------------------------------------------------------------------------
# DATASETS


@dataclass
class _DatasetsConfig:
    # Augmentation
    image_shape: tuple[int, int] = (240, 320)  #  # Image shape

    # Train configuration
    batch_size: int = 8  #                        # Training batch size
    num_workers: int = 16  #                      # Training number of workers
    path: str = "/data/datasets/"  #              # Training data path
    repeat: int = 1  #                            # Number of times training dataset is repeated per epoch


# --------------------------------------------------------------------------
# CONFIG


@dataclass(init=False)
class Config:
    name: str
    debug: bool
    arch: _ArchConfig
    wandb: _WandBConfig
    model: _ModelConfig
    datasets: _DatasetsConfig

    def __init__(
        self,
        name: str = "",
        debug: bool = True,
        arch: dict[str, Any] | _ArchConfig = _ArchConfig(),
        wandb: dict[str, Any] | _WandBConfig = _WandBConfig(),
        model: dict[str, Any] | _ModelConfig = _ModelConfig(),
        datasets: dict[str, Any] | _DatasetsConfig = _DatasetsConfig(),
    ) -> None:

        self.name = name  # Run name
        self.debug = debug  # Debugging flag

        self.arch = arch if isinstance(arch, _ArchConfig) else _ArchConfig(**arch)
        self.wandb = wandb if isinstance(wandb, _WandBConfig) else _WandBConfig(**wandb)
        self.model = model if isinstance(model, _ModelConfig) else _ModelConfig(**model)
        self.datasets = (
            datasets
            if isinstance(datasets, _DatasetsConfig)
            else _DatasetsConfig(**datasets)
        )

    @property
    def config(self) -> str:  # Run configuration file
        return ""

    @property
    def default(self) -> str:  # Run default configuration file
        return ""
