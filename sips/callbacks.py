"""
Implements PyTorch Lightning Callbacks for model training.

"""
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from sips.configs import Config


def _get_model_checkpoint_callback(
    monitor: str, mode: str, every_n_epochs: int
) -> ModelCheckpoint:
    """
    Return ModelCheckpoint callback.

    """
    filename = (
        "model-epoch={epoch}-step={step}"
        "-rep={val_repeatability:.4f}-mscore={val_matching_score:.4f}"
        "-loc={val_localization_error:.4f}"
    )
    return ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=2,
        auto_insert_metric_name=False,  # want to format 'filename' ourself
        filename=filename,
        every_n_epochs=every_n_epochs,  # plays hand in hand with EarlyStopping's patience
    )


def _get_early_stopping_callback(
    monitor: str, mode: str, patience: int
) -> EarlyStopping:
    """
    Return EarlyStopping callback.

    """
    return EarlyStopping(monitor=monitor, mode=mode, patience=patience)


def get_callbacks(monitors: list[tuple[str, str]], config: Config) -> list[Callback]:
    """
    Return list of training callbacks.

    Parameters
    ----------
    monitors : list[tuple[str, str]]
        List of (score_name, min_or_max) tuples. E.g., [("score_name", "min")].
    config : Config
        Moel configuration.

    Returns
    -------
    list[Callback]
        Training callbacks.

    """
    callbacks: list[Callback] = []

    # Learning rate monitor
    if not config.wandb.dry_run:
        callbacks.append(LearningRateMonitor())

    # Add checkpointing and stopping callback the given scores
    for monitor, mode in monitors:
        # Model checkpointing
        ckpt = _get_model_checkpoint_callback(
            monitor, mode, config.model.checkpoint_every_n_epochs
        )
        callbacks.append(ckpt)
        # Early stopping
        stop = _get_early_stopping_callback(
            monitor, mode, config.model.early_stopping_patience
        )
        callbacks.append(stop)

    return callbacks
