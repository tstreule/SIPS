"""
Script for SIPS training.

"""
from sips.utils.dotenv import load_dotenv

load_dotenv()


from typing import Optional

import torch._dynamo
import typer
from pytorch_lightning import Trainer, seed_everything
from typing_extensions import Annotated

from sips.callbacks import get_callbacks
from sips.configs import parse_train_file
from sips.datasets import SonarDataModule
from sips.models import KeypointNetwithIOLoss
from sips.utils.logging import setup_wandb_logger

app = typer.Typer()


def _set_debug(debug: bool):
    if not debug:
        return

    import torch
    from torch import autograd

    autograd.set_detect_anomaly(True)  # type: ignore[attr-defined]
    torch.set_printoptions(precision=2, threshold=100, edgeitems=2, sci_mode=False)
    torch._dynamo.config.verbose = True  # type: ignore[attr-defined]


def _torch_compile(
    model: KeypointNetwithIOLoss, accelerator: str, **kwargs
) -> KeypointNetwithIOLoss:
    if accelerator == "mps":
        # `torch.compile()` cannot handle MPS yet
        return model

    # NOTE: Ideally we would call `model = torch.compile(model)` and add a
    #  `@torch.compile(dynamic=True)` decorator to the
    #  `sips.utils.match_keypoints_2d_batch` function.
    #  However, due to a PyTorch issue this is currently not possible
    #  (see https://github.com/pytorch/pytorch/issues/99774).
    #  Therefore, we currently just compile the two models separately.
    torch._dynamo.reset()
    model.keypoint_net = torch.compile(model.keypoint_net, **kwargs)  # type: ignore[assignment]
    model.io_net = torch.compile(model.io_net, **kwargs)  # type: ignore[assignment]
    return model


@app.command()
def main(config_file: Annotated[Optional[str], typer.Option("--config")] = None):
    # Read configuration
    config_file = config_file or "sips/configs/v0_dummy.yaml"  # TODO: Delete this line
    config = parse_train_file(config_file)
    _set_debug(config.debug)

    # Initialize model and data module
    model = KeypointNetwithIOLoss.from_config(config.model)
    model = _torch_compile(model, config.arch.accelerator)
    dm = SonarDataModule(config.datasets)

    # Initialize callbacks
    callbacks = get_callbacks(
        monitors=[("val_repeatability", "max"), ("val_matching_score", "max")],
        config=config,
    )

    # Initialize trainer and make everything reproducible
    # c.f. https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
    seed_everything(config.arch.seed, workers=True)
    trainer = Trainer(
        deterministic=True,
        # Strategy
        strategy=config.arch.strategy,
        accelerator=config.arch.accelerator,
        devices=config.arch.devices,
        # Training args
        default_root_dir=config.model.checkpoint_path,
        enable_checkpointing=config.model.save_checkpoint,
        max_epochs=config.arch.max_epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=None,  # must be None since 'callbacks' is implemented
        logger=setup_wandb_logger(config),
        log_every_n_steps=config.arch.log_every_n_steps,
        # Debugging
        fast_dev_run=config.debug,
        num_sanity_val_steps=None,
    )

    # Train model
    trainer.fit(model, datamodule=dm)

    # Test (only right before publishing your paper or pushing to production!)
    # trainer.test(datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    app()
