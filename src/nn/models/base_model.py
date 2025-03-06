# """
# @author   Maksim Penkin
# """

from lightning import LightningModule

from . import nets
from src import nn
from src.utils.torch_utils import split_loss_logs


class CommonLitModel(LightningModule):

    def __init__(self, model, criterion=None, optimizer=None):
        super(CommonLitModel, self).__init__()

        self._model = nets.get(model)
        if criterion is not None:
            self._criterion = nn.losses.get(criterion)
        if optimizer is not None:
            self._optimizer = nn.optimizers.get(optimizer, partial=True)

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

    def training_step(self, batch, batch_idx, dataloader_idx):
        x, y = self.unpack_x_y(batch)
        y_pred = self(x)
        loss, logs = self.compute_loss(y_pred, y)
        self.log_dict(logs)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = self.unpack_x_y(batch)
        y_pred = self(x)
        loss, logs = self.compute_loss(y_pred, y)
        self.log_dict({"val_" + k: v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return self._optimizer(params=self.parameters())
