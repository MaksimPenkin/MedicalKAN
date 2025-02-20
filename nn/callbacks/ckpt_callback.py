# """
# @author   Maksim Penkin
# """

import os
from pathlib import Path
import torch
from utils.os_utils import make_dir

from nn.callbacks.base_callback import ICallback


class ModelCheckpointCallback(ICallback):

    @property
    def save_path(self):
        return self._save_path

    @property
    def save_freq(self):
        return self._save_freq

    @save_freq.setter
    def save_freq(self, value):
        if isinstance(value, int):
            assert value > 0
        elif value != "epoch":
            raise ValueError(f"Expected `save_freq` to be `epoch` or int, found: {value} of type {type(value)}.")
        self._save_freq = value

    def __init__(self, save_path, save_freq="epoch"):
        super(ModelCheckpointCallback, self).__init__()

        save_path = Path(save_path)
        make_dir(save_path.parent)

        self._save_path = save_path
        self.save_freq = save_freq

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq == "epoch":
            self._save_model(epoch, None)

    def on_train_batch_end(self, batch, logs=None):
        if self.save_freq == "epoch":
            return

        if (batch + 1) % self.save_freq == 0:
            self._save_model(self._current_epoch, batch)

    def _save_model(self, epoch, batch):
        with torch.no_grad():
            # `filepath` may contain placeholders such as `epoch` and `batch`.
            if batch is None:
                torch.save(self.model.state_dict(), os.fspath(self.save_path).format(epoch=epoch + 1))
            else:
                torch.save(self.model.state_dict(), os.fspath(self.save_path).format(epoch=epoch + 1, batch=batch + 1))
