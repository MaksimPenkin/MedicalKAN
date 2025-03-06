# """
# @author   Maksim Penkin
# """

import os, socket
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from .base_callback import ICallback


class TensorBoardCallback(ICallback):

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def update_freq(self):
        return self._update_freq

    @update_freq.setter
    def update_freq(self, value):
        if isinstance(value, int):
            assert value > 0
        elif value == "batch":
            value = 1
        elif value != "epoch":
            raise ValueError(f"Expected `update_freq` to be `epoch`, `batch` or int, found: {value} of type {type(value)}.")
        self._update_freq = value

    @property
    def global_step(self):
        return self._global_step

    def __init__(self, log_dir, update_freq="epoch", global_step=0):
        super(TensorBoardCallback, self).__init__()

        assert isinstance(global_step, int) and global_step >= 0
        self._global_step = global_step

        self._log_dir = Path(log_dir).joinpath(
            datetime.now().strftime("%b%d_%H-%M-%S") + "_" + socket.gethostname() if self.global_step == 0 else "",
            "logs")
        self._writers = {}

        self.update_freq = update_freq

    @property
    def _train_writer(self):
        if "train" not in self._writers:
            self._writers["train"] = SummaryWriter(
                log_dir=os.fspath(self.log_dir / "train")
            )
        return self._writers["train"]

    @property
    def _val_writer(self):
        if "val" not in self._writers:
            self._writers["val"] = SummaryWriter(
                log_dir=os.fspath(self.log_dir / "val")
            )
        return self._writers["val"]

    def _close_writers(self):
        for writer in self._writers.values():
            writer.close()

    def on_epoch_end(self, epoch, logs=None):
        self._log_epoch_metrics(epoch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self._log_train_batch_metrics(batch, logs)
        self._global_step += 1

    def on_train_end(self, logs=None):
        self._close_writers()

    def _log_epoch_metrics(self, epoch, logs):
        if not logs:
            return

        train_logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
        val_logs = {k: v for k, v in logs.items() if k.startswith("val_")}

        if train_logs:
            for name, value in train_logs.items():
                self._train_writer.add_scalar("epoch/" + name, value, global_step=epoch + 1)
        if val_logs:
            for name, value in val_logs.items():
                name = name[4:]  # Remove 'val_' prefix.
                self._val_writer.add_scalar("epoch/" + name, value, global_step=epoch + 1)

    def _log_train_batch_metrics(self, batch, logs):
        if not logs:
            return

        if self.update_freq == "epoch":
            return

        if (batch + 1) % self.update_freq == 0:
            for name, value in logs.items():
                self._train_writer.add_scalar("batch/" + name, value, global_step=self.global_step + 1)
