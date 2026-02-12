from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Sequence, TypeVar

from ..config import ExperimentTrainConfig
from ..data import DatasetConfig, Split, get_dataset_configs
from ..init import ExperimentInitConfig
from ..losses import get_criterion
from ..metadata import DataManager
from ..models import get_training_models
from ..utilities import comm


class ZipDataloader:
    """Zips multiple dataloaders together."""

    def __init__(self, dataloaders: list[Sequence]) -> None:
        self.dataloaders = dataloaders
        self.length = min(len(dl) for dl in dataloaders)

    def __len__(self):
        return self.length

    def __iter__(self):
        return zip(*self.dataloaders)

    def __getitem__(self, idx):
        return [dl[idx] for dl in self.dataloaders]


def _get_dataloader_from_dataset_configs(
    configs: list[DatasetConfig], split: Split
) -> Sequence:
    """Normalize a multiple dataloaders to a ZipDataloader or return single dataloader"""
    dataloaders = [cfg.get_dataloader(split) for cfg in configs]
    if len(dataloaders) > 1:
        return ZipDataloader(dataloaders)
    return dataloaders[0]


@dataclass
class TrainerModules:
    """Holds all common training Modules"""

    # Model to train, unwrap from list, ideally this is always one module
    # with multiple submodule if necessary, like a GAN
    model: Any
    criterion: list[Any]  # list of loss functions
    optimizer: list[Any]  # Optimizer
    scheduler: list[Any]  # Learning rate scheduler
    trainloader: Sequence
    valloader: Sequence | None

    @classmethod
    def from_init_config(cls, exp_config: ExperimentInitConfig):
        """Instantiate training modules from ExperimentInitConfig"""
        dataset_cfgs = get_dataset_configs(exp_config)
        train_loaders = _get_dataloader_from_dataset_configs(dataset_cfgs, Split.TRAIN)
        val_loaders = _get_dataloader_from_dataset_configs(dataset_cfgs, Split.VAL)
        models, optims, scheds = get_training_models(exp_config)
        criterion = get_criterion(exp_config)
        return cls(models, criterion, optims, scheds, train_loaders, val_loaders)

    @classmethod
    def from_train_config(cls, cfg: ExperimentTrainConfig):
        """Instantiate training modules from ExperimentTrainConfig"""
        # Get train and val dataloaders
        train_loaders = _get_dataloader_from_dataset_configs(cfg.dataset, Split.TRAIN)
        val_loaders = _get_dataloader_from_dataset_configs(cfg.dataset, Split.VAL)
        models, optims, scheds = cfg.get_all_training_modules()
        criterion = [c.get_instance() for c in cfg.criterion]
        return cls(models, criterion, optims, scheds, train_loaders, val_loaders)

    def __post_init__(self):
        # Remove list wrapper if only one model for simplicity
        # otherwise the downstream should convert to the native
        # framework module list i.e. nn.ModuleList
        if isinstance(self.model, list) and len(self.model) == 1:
            self.model = self.model[0]

        # Normalize modules to list of one
        for mod in ["criterion", "optimizer", "scheduler"]:
            val = getattr(self, mod)
            if not isinstance(val, list):
                setattr(self, mod, [val])

    def get_checkpointables(self):
        """
        Get dictionary of training modules which typically include
        a state_dict that should be checkpointed during training
        i.e. model, optimizer and scheduler.
        """
        return {
            "model": self.model,
            "optim": self.optimizer,
            "scheduler": self.scheduler,
        }


@dataclass
class TrainerConfig:
    # Function to run for monitoring issues with the value
    # of the loss, does absolutely nothing by default
    loss_monitor: Callable[[dict[str, Any]], None] = lambda x: None

    # Enable Console Progress
    pbar: Callable | None = None

    # Run evaluation before beginning of training
    pre_eval: bool = False

    # Train for specified iterations then validate
    validation_interval: int | None = None

    def __post_init__(self):
        if comm.get_local_rank() != 0:
            self.pbar = None  # Ensure only one pbar per machine


class TrainingError(RuntimeError):
    """Exception raised by user in their training loop"""


class BaseTrainer(ABC):
    """
    Base class that various trainer types inherit from that
    contains basic train loops which they can implement
    """

    modules: TrainerModules

    def __init__(
        self, config: TrainerConfig, modules: TrainerModules, data_manager: DataManager
    ):
        self.modules = modules
        self.data_manager = data_manager
        self._logger = getLogger(type(self).__name__)
        self._config = config
        self.data_manager.resume()

        self.pre_train_hooks: list[Callable] = []
        self.post_train_hooks: list[Callable] = []
        self.pre_val_hooks: list[Callable] = []
        self.post_val_hooks: list[Callable] = []

        if config.pbar is not None:
            train_len = len(self.modules.trainloader)
            if config.validation_interval is not None:
                train_len = min(train_len, config.validation_interval)

            self._train = config.pbar(self._train, total=train_len, desc="Training")
            if self.modules.valloader is not None:
                self._validate = config.pbar(
                    self._validate, total=len(self.modules.valloader), desc="Validation"
                )

    def run_epoch(self, max_iter: int | None = None) -> None:
        """Complete one epoch with training and validation epoch"""
        _run_hooks(self.pre_train_hooks)
        self._train(max_iter)
        _run_hooks(self.post_train_hooks)

        val_interval = self._config.validation_interval
        if val_interval is None or self.data_manager.iteration % val_interval == 0:
            _run_hooks(self.pre_val_hooks)
            self._validate()
            _run_hooks(self.post_val_hooks)

        for sched in self.modules.scheduler:
            self._maybe_step_scheduler(sched, is_epoch=True)
        self.data_manager.epoch_step()

    def train(self, *, epoch: int | None = None, iteration: int | None = None) -> None:
        """Train until epoch or iteration is reached, keyword only to prevent bugs"""
        if self._config.pre_eval and self.data_manager.iteration == 0:
            _run_hooks(self.pre_val_hooks)
            self._validate()
            _run_hooks(self.post_val_hooks)

        if iteration is not None:
            assert epoch is None, "Only epoch or iteration should be specified"
            if self.data_manager.ckpt_cfg.epoch_mode:
                self._logger.warning(
                    "Checkpointer in epoch mode but training in iteration mode"
                )

            while self.data_manager.iteration < iteration:
                self._logger.info(
                    "Training %d of %d iterations",
                    self.data_manager.iteration,
                    iteration,
                )
                self.run_epoch(iteration)
        else:
            assert epoch is not None, "Neither epoch or iteration were specified"
            if self.data_manager.ckpt_cfg.iter_mode:
                self._logger.warning(
                    "Checkpointer in iteration mode but training in epoch mode"
                )

            while self.data_manager.epoch < epoch:
                self._logger.info(
                    "Training %d of %d epochs", self.data_manager.epoch, epoch
                )
                self.run_epoch(iteration)

        self._logger.info("Finished Training, Saving Model and Metadata")
        self.data_manager.save("latest", force_push=True)
        self._logger.info("Finished Saving (and Pushing)")

    def data_transform(self, data: Any) -> Any:
        """
        Apply any post motifications to data after loading before
        being passed to [train|val]_step, no-op by default
        """
        return data

    def training_exception(self, err: Exception, data: Any) -> None:
        """
        This function is run when an runtime exception is thrown during
        training iteration, useful for logging the state of the model
        and the data used in the training iteration.
        """
        raise err

    def _should_break_training_loop(self, max_iter: int | None):
        """
        Check whether to break out of the training loop if the target
        maximum iteration is reached, or validation should be run.
        """
        cond = False
        cur_iter = self.data_manager.iteration
        if max_iter is not None:
            cond |= max_iter <= cur_iter
        if val_interval := self._config.validation_interval:
            cond |= cur_iter % val_interval == 0 and cur_iter > 0
        return cond

    @abstractmethod
    def _accumulate_losses(self, losses: dict[str, Any]) -> Any:
        """Accumulate losses into single number hook, good idea to put a
        grad scaler here if using amp"""

    @abstractmethod
    def _maybe_step_optimiser(self, optim: Any, sched: Any) -> bool:
        """Step optimizer if iteration is divisible by interval

        Returns:
            True if the optimizer was stepped
        """

    @abstractmethod
    def _maybe_step_scheduler(self, sched: Any, is_epoch: bool) -> bool:
        """Step lr scheduler if necessary

        Returns:
            True if the scheduler was stepped
        """

    @abstractmethod
    def _train(self, max_iter: int | None) -> None:
        """Train for one epoch over the dataset or to the
        optional global iteration limit"""

    @abstractmethod
    def _validate(self) -> None:
        """Validate one epoch over the dataset"""


def _run_hooks(hooks: list[Callable]):
    for h in hooks:
        h()


TrainerT = TypeVar("TrainerT", bound=BaseTrainer)
