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
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return self._optimizer(params=self.parameters())
