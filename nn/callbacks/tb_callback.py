# """
# @author   Maksim Penkin
# """

import os, socket
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from nn.callbacks.base_callback import ICallback


class TensorBoardCallback(ICallback):

    @property
    def global_step(self):
        return self._global_step

    @property
    def update_freq(self):
        return self._update_freq

    @update_freq.setter
    def update_freq(self, value):
        if value == "batch":
            value = 1
        elif isinstance(value, int):
            assert value > 0
        elif value != "epoch":
            raise ValueError("nn/callbacks/tb_callback.py: class TensorBoardCallback: @update_freq.setter: "
                             f"error: expected `update_freq` to be `epoch`, `batch` or int, found: {value} of type {type(value)}.")
        self._update_freq = value

    def __init__(self, log_dir=None, update_freq="epoch"):
        super(TensorBoardCallback, self).__init__()

        log_dir = os.path.join(log_dir or "runs", datetime.now().strftime("%b%d_%H-%M-%S") + "_" + socket.gethostname())
        self.log_dir = os.path.join(log_dir, "logs")

        self.update_freq = update_freq

        # Lazily initialized in order to avoid creating event files when not needed.
        self._writers = {}
        self._global_step = 0

    def set_model(self, model):
        super(TensorBoardCallback, self).set_model(model)

        self._writers = {}
        self._global_step = 0

    @property
    def _train_writer(self):
        if "train" not in self._writers:
            self._writers["train"] = SummaryWriter(
                log_dir=os.path.join(self.log_dir, "train")
            )
        return self._writers["train"]

    @property
    def _val_writer(self):
        if "val" not in self._writers:
            self._writers["val"] = SummaryWriter(
                log_dir=os.path.join(self.log_dir, "val")
            )
        return self._writers["val"]

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

    def _close_writers(self):
        for writer in self._writers.values():
            writer.close()
