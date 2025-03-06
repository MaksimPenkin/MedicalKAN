# """
# @author   Maksim Penkin
# """

from .base_callback import ICallback
from .ckpt_callback import ModelCheckpointCallback
from .tb_callback import TensorBoardCallback

from utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "ModelCheckpointCallback": ModelCheckpointCallback,
                            "TensorBoardCallback": TensorBoardCallback,
                            "CompositeCallback": CompositeCallback},
                        **kwargs)

    if isinstance(obj, ICallback):
        return obj
    raise ValueError(f"Could not interpret callback instance: {obj}.")


class CompositeCallback(ICallback):
    """Container abstracting a list of callbacks."""

    def __init__(self, callbacks=None, model=None):
        """Container for `ICallback` instances.

        This object wraps a list of `ICallback` instances, making it possible
        to call them all at once via a single endpoint
        (e.g. `callback_list.on_epoch_end(...)`).

        Args:
            callbacks: List of `ICallback` instances.
        """
        super(CompositeCallback, self).__init__()

        if callbacks:
            self.callbacks = [get(c) for c in callbacks]
        else:
            self.callbacks = []

        if model:
            self.set_model(model)

    def set_model(self, model):
        super(CompositeCallback, self).set_model(model)

        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_batch_begin(batch, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_batch_end(batch, logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_batch_begin(batch, logs=logs)

    def on_predict_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_end(logs)
