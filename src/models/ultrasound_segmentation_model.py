# """
# @author   Maksim Penkin
# """

from .base_model import CommonLitModel


class UltrasoundSegmentationModel(CommonLitModel):
    def __init__(self, *args, **kwargs):
        super(UltrasoundSegmentationModel, self).__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_pred = self(x)
        loss, logs = self.compute_loss(y_pred, y)
        logs = {"train/" + k: v for k, v in logs.items()}
        self.log_dict(logs, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_pred = self(x)
        loss, logs = self.compute_loss(y_pred, y)
        logs = {"val/" + k: v for k, v in logs.items()}
        logs["step"] = self.current_epoch
        self.log_dict(logs, on_step=False, on_epoch=True)
        return loss
