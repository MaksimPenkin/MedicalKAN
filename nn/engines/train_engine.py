# """
# @author   Maksim Penkin
# """

import torch
from tqdm import tqdm

import data
from nn import losses, optimizers
from nn.callbacks import CompositeCallback
from metrics import CompositeMetric

from utils.torch_utils import split_loss_logs, to_device

from nn.engines.base_engine import IEngine


class Trainer(IEngine):

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataloader(self):
        return self._eval_dataloader

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def callbacks(self):
        return self._callbacks

    def __init__(self, dataloader, criterion, *args, optimizer="adam", callbacks=None, val_dataloader=None, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

        self._train_dataloader = data.get(dataloader)
        if val_dataloader is not None:
            self._eval_dataloader = data.get(val_dataloader)
        else:
            self._eval_dataloader = None

        self._criterion = losses.get(criterion)
        self._optimizer = optimizers.get(optimizer, params=self.model.parameters())
        self._callbacks = CompositeCallback(callbacks=callbacks, model=self.model)

    @staticmethod
    def unpack_x_y(blob, device="cpu"):
        x, y = blob
        return to_device(x, device=device), to_device(y, device=device)

    def compute_loss(self, y_pred, y):
        if isinstance(y, dict):
            try:
                value = self.criterion(y_pred, **y)
            except:
                value = self.criterion(y_pred, y)
        elif isinstance(y, (list, tuple)):
            try:
                value = self.criterion(y_pred, *y)
            except:
                value = self.criterion(y_pred, y)
        else:
            value = self.criterion(y_pred, y)

        return split_loss_logs(value)

    def train_step(self, blob, device="cpu"):
        x, y = self.unpack_x_y(blob, device=device)

        self.optimizer.zero_grad()
        y_pred = self.model_step(x)
        loss, logs = self.compute_loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return logs

    def eval_step(self, blob, device="cpu"):
        x, y = self.unpack_x_y(blob, device=device)

        with torch.no_grad():
            y_pred = self.model_step(x)
            _, logs = self.compute_loss(y_pred, y)
        return {"val_" + k: v for k, v in logs.items()}

    def fit(self, epochs=1, limit_batches=1.0, device="cpu"):
        steps = int(limit_batches * len(self.train_dataloader))
        val_steps = len(self.eval_dataloader) if self.eval_dataloader is not None else None

        train_tracker = CompositeMetric()
        val_tracker = CompositeMetric() if self.eval_dataloader is not None else None

        self.model.to(device)
        self.callbacks.on_train_begin()
        epoch_logs = {}
        for epoch in tqdm(range(epochs)):
            train_tracker.reset_state()
            self.model.train()
            self.callbacks.on_epoch_begin(epoch)
            for idx, blob in enumerate(tqdm(self.train_dataloader, total=steps, leave=False)):
                if idx >= steps:
                    break
                self.callbacks.on_train_batch_begin(idx)
                logs = self.train_step(blob, device=device)
                self.callbacks.on_train_batch_end(idx, logs=logs)
                train_tracker.update_state(logs, n=blob.size(0))  # TODO: add seamless batch_size value extraction.
            epoch_logs = train_tracker.result()

            if self.eval_dataloader is not None:
                val_tracker.reset_state()
                self.model.eval()
                self.callbacks.on_test_begin()
                for idx, blob in enumerate(tqdm(self.eval_dataloader, total=val_steps, leave=False)):
                    if idx >= val_steps:
                        break
                    self.callbacks.on_test_batch_begin(idx)
                    logs = self.eval_step(blob, device=device)
                    self.callbacks.on_test_batch_end(idx, logs=logs)
                    val_tracker.update_state(logs, n=blob.size(0))  # TODO: add seamless batch_size value extraction.
                val_logs = val_tracker.result()
                self.callbacks.on_test_end(logs=val_logs)
                epoch_logs.update(val_logs)

            self.callbacks.on_epoch_end(epoch, logs=epoch_logs)
        self.callbacks.on_train_end(logs=epoch_logs)
