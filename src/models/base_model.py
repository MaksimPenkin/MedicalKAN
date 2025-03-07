# """
# @author   Maksim Penkin
# """

from . import nets
from ..nn import losses, optimizers
from src.utils.torch_utils import split_loss_logs

from lightning import LightningModule


class CommonLitModel(LightningModule):
    def __init__(self, model, criterion=None, optimizer=None):
        super(CommonLitModel, self).__init__()

        self._model = nets.get(model)
        if criterion is not None:
            self._criterion = losses.get(criterion)
        if optimizer is not None:
            self._optimizer = optimizers.get(optimizer, partial=True)

    @staticmethod
    def unpack_x_y(batch):
        x, y = batch
        return x, y

    def forward(self, x, **kwargs):
        if isinstance(x, dict):
            try:
                y_pred = self._model(**x, **kwargs)
            except:
                y_pred = self._model(x, **kwargs)
        elif isinstance(x, (list, tuple)):
            try:
                y_pred = self._model(*x, **kwargs)
            except:
                y_pred = self._model(x, **kwargs)
        else:
            y_pred = self._model(x, **kwargs)

        return y_pred

    def compute_loss(self, y_pred, y):
        if isinstance(y, dict):
            try:
                value = self._criterion(y_pred, **y)
            except:
                value = self._criterion(y_pred, y)
        elif isinstance(y, (list, tuple)):
            try:
                value = self._criterion(y_pred, *y)
            except:
                value = self._criterion(y_pred, y)
        else:
            value = self._criterion(y_pred, y)

        return split_loss_logs(value)

    def training_step(self, batch, batch_idx):
        x, y = self.unpack_x_y(batch)
        y_pred = self(x)
        loss, logs = self.compute_loss(y_pred, y)
        logs = {"train/" + k: v for k, v in logs.items()}
        self.log_dict(logs, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.unpack_x_y(batch)
        y_pred = self(x)
        loss, logs = self.compute_loss(y_pred, y)
        logs = {"val/" + k: v for k, v in logs.items()}
        logs["step"] = self.current_epoch
        self.log_dict(logs, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return self._optimizer(params=self.parameters())
