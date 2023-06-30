import os
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# ARCH


@dataclass
class _ArchConfig:
    seed: int = 42  #                        # Random seed

    # Strategy
    strategy: str = "auto"  #                # ("ddp", ...)
    accelerator: str = "auto"  #             # ("cpu", "gpu", "tpu", ..., "auto")
    devices: str | int = "auto"  #           # The devices to use
    precision: str | int = "32-true"  #      # precision

    # Training args
    max_epochs: int = 50  #                  # Maximum number of epochs
    fast_dev_run: bool = False  #            # Enable for debugging purposes


# --------------------------------------------------------------------------
# WANDB


@dataclass
class _WandBConfig:
    dry_run: bool = True  #                        # If True, do a dry run (no logging)

    offline: bool = False  #                       # Run offline (data can be streamed later to WandB servers)
    name: str = ""  #                              # Display name for the run.
    save_dir: str = ""  #                          # Path where data is saved
    version: str = ""  #                           # WandB version, to resume prev. run
    project_os_env: str = "WANDB_PROJECT"  #       # WandB project (from os.env)
    entity_os_env: str = "WANDB_ENTITY"  #         # WandB entity (from os.env)
    tags: list[str] = field(default_factory=list)  # WandB tags

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
    seed: int = 42  #                        # Random seed

    # Rosbags to consider
    rosbags: list[str] = field(default_factory=list)

    # Augmentation
    image_shape: tuple[int, int] = (512, 512)  #  # Image shape

    # Redundant Image Filter configuration
    image_filter: str | None = "bilateral"  # Type of image filter
    image_filter_size: float = 5  # Filter size
    image_filter_std: float = 2.5  # Standard deviation for Gaussian filter
    image_filter_threshold: float = (
        0.95  # Similarity threshold, higher values results in more data
    )

    # Sonar configuration
    horizontal_fov: float = 130  # Horizontal field of view
    vertical_fov: float = 20  # Vertical field of view
    max_range: float = 40  # Max range for sonar
    min_range: float = 0.1  # Min range for sonar

    # Image overlap configuration
    bright_spots_threshold: int = 200  # Threshold to detect bright spots, higher values results in less bright spots
    overlap_ratio_threshold: float = (
        0.5  # Ratio of image overlap averaged over symmetric comparison
    )
    n_elevations: int = 5  # Number of elevations for arc projections
    conv_size: int = (
        8  # Convolution size for bright spots detection and arc projections
    )

    train_ratio: float = 0.8

    # Train configuration
    batch_size: int = 8  #                        # Training batch size
    num_workers: int = 8  #                       # Training number of workers
    path: str = "/data/datasets/"  #              # Training data path
    repeat: int = 1  #                            # Number of times training dataset is repeated per epoch

    @property
    def val_ratio(self) -> float:
        return 1 - self.train_ratio

    # Return all parameters of this config that can be tuned
    def get_variable_params(self) -> dict[str, float | int | str | None]:
        variable_params = {
            "image_filter": self.image_filter,
            "image_filter_size": self.image_filter_size,
            "image_filter_std": self.image_filter_std,
            "image_filter_threshold": self.image_filter_threshold,
            "bright_spots_threshold": self.bright_spots_threshold,
            "overlap_ratio_threshold": self.overlap_ratio_threshold,
            "n_elevations": self.n_elevations,
        }
        return variable_params


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
