# """
# @author   Maksim Penkin
# """

class ICallback:
    """Base class used to build new callbacks.

    To create a custom callback, subclass `ICallback` and
    override the method associated with the stage of interest.
    """

    @property
    def model(self):
        return self._model

    def __init__(self):
        self._model = None

    def set_model(self, model):
        self._model = model

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_test_batch_end()` is passed to this argument for this method
              but that may change in the future.
        """

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_predict_end(self, logs=None):
        """Called at the end of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """


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
            assert all(isinstance(c, ICallback) for c in callbacks), f"Error: expected all elements of `callbacks` ICallback, found: {callbacks}."
            self.callbacks = [callback for callback in callbacks]
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
