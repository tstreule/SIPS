from dataclasses import asdict

from pytorch_lightning import Trainer, seed_everything

from sips.configs import parse_train_file
from sips.datasets import SonarDataModule


def main():
    # Read configuration
    config = parse_train_file("sips/configs/v0_dummy.yaml")

    # Initialize trainer and make everything reproducible
    # c.f. https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
    seed_everything(config.arch.seed, workers=True)
    trainer = Trainer(deterministic=True)

    # Initialize model and data module
    model = Model(**asdict(config.model.params))
    dm = SonarDataModule(config.datasets)

    # Train model
    trainer.fit(model, datamodule=dm, ckpt_path=None)

    # Test (only right before publishing your paper or pushing to production!)
    # trainer.test(datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
