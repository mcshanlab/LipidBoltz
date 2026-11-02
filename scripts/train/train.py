import os
import random
import signal
import string
import sys
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from omegaconf import OmegaConf, listconfig
from pytorch_lightning import LightningModule
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from boltz.data.module.training import BoltzTrainingDataModule, DataConfig


@dataclass
class TrainConfig:
    """Train configuration.

    Attributes
    ----------
    data : DataConfig
        The data configuration.
    model : ModelConfig
        The model configuration.
    output : str
        The output directory.
    trainer : Optional[dict]
        The trainer configuration.
    resume : Optional[str]
        The resume checkpoint.
    pretrained : Optional[str]
        The pretrained model.
    wandb : Optional[dict]
        The wandb configuration.
    disable_checkpoint : bool
        Disable checkpoint.
    matmul_precision : Optional[str]
        The matmul precision.
    find_unused_parameters : Optional[bool]
        Find unused parameters.
    save_top_k : Optional[int]
        Save top k checkpoints.
    checkpoint_every_n_mins : Optional[int]
        Save checkpoints every N minutes if provided.
    validation_only : bool
        Run validation only.
    debug : bool
        Debug mode.
    strict_loading : bool
        Fail on mismatched checkpoint weights.
    load_confidence_from_trunk: Optional[bool]
        Load pre-trained confidence weights from trunk.

    """

    data: DataConfig
    model: LightningModule
    output: str
    trainer: Optional[dict] = None
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    wandb: Optional[dict] = None
    disable_checkpoint: bool = False
    matmul_precision: Optional[str] = None
    find_unused_parameters: Optional[bool] = False
    save_top_k: Optional[int] = 1
    checkpoint_every_n_mins: Optional[int] = None
    validation_only: bool = False
    debug: bool = False
    strict_loading: bool = True
    load_confidence_from_trunk: Optional[bool] = False
    run_pre_validation: bool = True


class GracefulExitCheckpoint(Callback):
    """Save a final checkpoint on preemption-like signals and stop cleanly."""

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir
        self._should_stop = False
        self._saved = False

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        del trainer, pl_module, stage
        signal.signal(signal.SIGUSR1, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: object) -> None:
        del frame
        print(f"Received signal {signum}; will save checkpoint and stop.")
        self._should_stop = True

    def _save_and_stop(self, trainer: pl.Trainer) -> None:
        if self._saved:
            trainer.should_stop = True
            return
        checkpoints_dir = Path(self.output_dir) / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoints_dir / "preempt_last.ckpt"
        print(f"Saving graceful-exit checkpoint to {ckpt_path}")
        trainer.save_checkpoint(str(ckpt_path))
        self._saved = True
        # Lightning treats ``should_stop`` as a reason to validate the current
        # epoch, even when it is only partially complete. Suppress that
        # one-off validation so the resumed run's next validation remains on
        # the configured schedule.
        trainer.fit_loop.epoch_loop._skip_next_val = True  # noqa: SLF001
        trainer.should_stop = True

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        if self._should_stop:
            self._save_and_stop(trainer)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if self._should_stop:
            self._save_and_stop(trainer)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del pl_module, outputs, batch, batch_idx, dataloader_idx
        if self._should_stop:
            self._save_and_stop(trainer)


class CompletedGlobalStepCheckpoint(Callback):
    """Save a checkpoint when training reaches its configured maximum step."""

    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = output_dir

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        max_steps = trainer.max_steps
        global_step = trainer.global_step
        if max_steps <= 0 or global_step < max_steps:
            return

        checkpoints_dir = Path(self.output_dir) / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoints_dir / f"complete_global_step:{global_step}.ckpt"
        print(
            "Training reached the configured maximum step; "
            f"saving completion checkpoint to {ckpt_path}"
        )
        trainer.save_checkpoint(str(ckpt_path))


class FilteredWandbLogger(WandbLogger):
    """W&B logger that drops noisy framework metrics."""

    train_norm_every_n_optimizer_steps: int = 1

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        metrics = dict(metrics)
        metrics.pop("trainer/global_step", None)
        frequency = self.train_norm_every_n_optimizer_steps
        if frequency == 0 or step is None or (step + 1) % frequency != 0:
            metrics = {
                name: value
                for name, value in metrics.items()
                if not name.startswith("train_norm/")
            }
        super().log_metrics(metrics, step=step)


class FullValidationModelCheckpoint(ModelCheckpoint):
    """Skip monitored checkpoint updates during fast validation passes."""

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: LightningModule,
    ) -> None:
        data_module = trainer.datamodule
        if hasattr(data_module, "is_fast_validation"):
            if data_module.is_fast_validation():
                return
        super().on_validation_end(trainer, pl_module)


def train(raw_config: str, args: list[str]) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run training.

    Parameters
    ----------
    raw_config : str
        The input yaml configuration.
    args : list[str]
        Any command line overrides.

    """
    # Load the configuration
    raw_config = omegaconf.OmegaConf.load(raw_config)

    # Apply input arguments
    args = omegaconf.OmegaConf.from_dotlist(args)
    raw_config = omegaconf.OmegaConf.merge(raw_config, args)

    # Instantiate the task
    cfg = hydra.utils.instantiate(raw_config)
    cfg = TrainConfig(**cfg)

    # Set matmul precision
    if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Create trainer dict
    trainer = cfg.trainer
    if trainer is None:
        trainer = {}

    full_val_check_interval = trainer.get("val_check_interval")
    small_val_check_interval = trainer.pop("small_val_check_interval", None)
    if small_val_check_interval is not None:
        if full_val_check_interval is None:
            msg = "small_val_check_interval requires val_check_interval"
            raise ValueError(msg)
        if full_val_check_interval % small_val_check_interval != 0:
            msg = "val_check_interval must be divisible by small_val_check_interval"
            raise ValueError(msg)
        trainer["val_check_interval"] = small_val_check_interval

    # Flip some arguments in debug mode
    devices = trainer.get("devices", 1)

    wandb = cfg.wandb
    if cfg.debug:
        if isinstance(devices, int):
            devices = 1
        elif isinstance(devices, (list, listconfig.ListConfig)):
            devices = [devices[0]]
        trainer["devices"] = devices
        cfg.data.num_workers = 0
        if wandb:
            wandb = None

    # Create objects
    data_config = DataConfig(**cfg.data)
    data_module = BoltzTrainingDataModule(data_config)
    data_module.configure_validation_intervals(
        full_interval=full_val_check_interval,
        small_interval=small_val_check_interval,
    )
    model_module = cfg.model

    if cfg.pretrained and not cfg.resume:
        # Load the pretrained weights into the confidence module
        if cfg.load_confidence_from_trunk:
            checkpoint = torch.load(cfg.pretrained, map_location="cpu")

            # Modify parameter names in the state_dict
            new_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("structure_module") and not key.startswith(
                    "distogram_module"
                ):
                    new_key = "confidence_module." + key
                    new_state_dict[new_key] = value
            new_state_dict.update(checkpoint["state_dict"])

            # Update the checkpoint with the new state_dict
            checkpoint["state_dict"] = new_state_dict

            # Save the modified checkpoint
            random_string = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=10)
            )
            file_path = os.path.dirname(cfg.pretrained) + "/" + random_string + ".ckpt"
            print(
                f"Saving modified checkpoint to {file_path} created by broadcasting trunk of {cfg.pretrained} to confidence module."
            )
            torch.save(checkpoint, file_path)
        else:
            file_path = cfg.pretrained

        print(f"Loading model from {file_path}")
        model_module = type(model_module).load_from_checkpoint(
            file_path, map_location="cpu", strict=False, **(model_module.hparams)
        )

        if cfg.load_confidence_from_trunk:
            os.remove(file_path)

    # Create checkpoint callback
    callbacks = [
        GracefulExitCheckpoint(cfg.output),
        CompletedGlobalStepCheckpoint(cfg.output),
    ]
    dirpath = cfg.output
    if not cfg.disable_checkpoint:
        checkpoint_dir = Path(cfg.output) / "checkpoints"

        # Disabled: preemption recovery is handled by GracefulExitCheckpoint,
        # so do not create additional last_epoch*.ckpt snapshots.
        # epoch_last_checkpoint = ModelCheckpoint(
        #     dirpath=str(checkpoint_dir),
        #     monitor=None,
        #     save_top_k=0,
        #     save_last=True,
        #     every_n_epochs=1,
        #     every_n_train_steps=None,
        #     train_time_interval=None,
        #     save_on_train_epoch_end=True,
        # )
        # epoch_last_checkpoint.CHECKPOINT_NAME_LAST = "last_epoch"
        # callbacks.append(epoch_last_checkpoint)

        # Disabled: do not select checkpoints using confidence-ranked ligand
        # success. Keep this configuration here for reference.
        # best_ligand_checkpoint = FullValidationModelCheckpoint(
        #     dirpath=str(checkpoint_dir),
        #     monitor="val/top1_confidence_ligand_successrate",
        #     save_top_k=cfg.save_top_k,
        #     save_last=False,
        #     mode="max",
        #     every_n_epochs=1,
        #     every_n_train_steps=None,
        #     train_time_interval=None,
        #     filename=(
        #         "best-step={step:09d}"
        #         "-success={val/top1_confidence_ligand_successrate:.4f}"
        #     ),
        #     auto_insert_metric_name=False,
        # )
        # callbacks.append(best_ligand_checkpoint)

        # Keep the three checkpoints with the highest oracle best-of-samples
        # ligand success rate from completed full validation passes.
        best_ligand_oracle_checkpoint = FullValidationModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val/ligand_successrate_best",
            save_top_k=3,
            save_last=False,
            mode="max",
            every_n_epochs=1,
            every_n_train_steps=None,
            train_time_interval=None,
            filename=(
                "step={step:09d}"
                "-bestsuccess={val/ligand_successrate_best:.4f}"
            ),
            auto_insert_metric_name=False,
        )
        callbacks.append(best_ligand_oracle_checkpoint)

        # Independently keep the three checkpoints with the highest
        # ligand-protein lDDT from completed full validation passes.
        best_ligand_protein_lddt_checkpoint = FullValidationModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val/lddt_ligand_protein",
            save_top_k=3,
            save_last=False,
            mode="max",
            every_n_epochs=1,
            every_n_train_steps=None,
            train_time_interval=None,
            filename=(
                "step={step:09d}"
                "-ligandproteinlddt={val/lddt_ligand_protein:.4f}"
            ),
            auto_insert_metric_name=False,
        )
        callbacks.append(best_ligand_protein_lddt_checkpoint)

        # Optional time-based checkpointing that keeps only the latest timed snapshot.
        if cfg.checkpoint_every_n_mins is not None:
            timed_checkpoint = ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                monitor=None,
                save_top_k=1,
                save_last=False,
                mode="max",
                every_n_epochs=None,
                every_n_train_steps=None,
                train_time_interval=timedelta(minutes=cfg.checkpoint_every_n_mins),
                filename="timer-{epoch:04d}-{step:09d}",
            )
            callbacks.append(timed_checkpoint)

    # Create wandb logger
    loggers = []
    if wandb:
        wdb_logger = FilteredWandbLogger(
            name=wandb["name"],
            group=wandb["name"],
            save_dir=cfg.output,
            project=wandb["project"],
            entity=wandb["entity"],
            log_model=False,
        )
        wdb_logger.train_norm_every_n_optimizer_steps = getattr(
            model_module,
            "train_norm_every_n_optimizer_steps",
            1,
        )
        loggers.append(wdb_logger)
        # Save the config to wandb

        @rank_zero_only
        def save_config_to_wandb() -> None:
            config_out = Path(wdb_logger.experiment.dir) / "run.yaml"
            with Path.open(config_out, "w") as f:
                OmegaConf.save(raw_config, f)
            wdb_logger.experiment.save(str(config_out))

        save_config_to_wandb()

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, (list, listconfig.ListConfig)) and len(devices) > 1
    ):
        strategy = DDPStrategy(find_unused_parameters=cfg.find_unused_parameters)

    if not cfg.validation_only:
        trainer.setdefault("num_sanity_val_steps", 0)

    trainer = pl.Trainer(
        default_root_dir=str(dirpath),
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=not cfg.disable_checkpoint,
        reload_dataloaders_every_n_epochs=1,
        **trainer,
    )

    if not cfg.strict_loading:
        model_module.strict_loading = False

    if cfg.validation_only:
        trainer.validate(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume,
        )
    else:
        if cfg.run_pre_validation:
            if hasattr(model_module, "validation_log_prefix"):
                model_module.validation_log_prefix = "pre_val"
            trainer.validate(
                model_module,
                datamodule=data_module,
                ckpt_path=cfg.resume,
            )
        if hasattr(model_module, "validation_log_prefix"):
            model_module.validation_log_prefix = "val"
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume,
        )


if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2:]
    train(arg1, arg2)
