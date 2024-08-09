# """
# @author   Maksim Penkin
# """

import os
import torch
from utils.os_utils import create_folder

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
            raise ValueError("nn/callbacks/ckpt_callback.py: class ModelCheckpointCallback: @save_freq.setter: "
                             f"error: expected `save_freq` to be `epoch` or int, found: {value} of type {type(value)}.")
        self._save_freq = value

    def __init__(self, save_path, save_freq="epoch"):
        super(ModelCheckpointCallback, self).__init__()

        save_dir = os.path.split(save_path)[0]
        if save_dir:  # e.g. os.path.split("name.png") -> '', 'name.png'; os.path.split("./name.png") -> '.', 'name.png'
            create_folder(save_dir)
        self._save_path = save_path

        self.save_freq = save_freq

    def set_model(self, model):
        super(ModelCheckpointCallback, self).set_model(model)

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
                torch.save(self.model.state_dict(), self.save_path.format(epoch=epoch + 1))
            else:
                torch.save(self.model.state_dict(), self.save_path.format(epoch=epoch + 1, batch=batch + 1))
