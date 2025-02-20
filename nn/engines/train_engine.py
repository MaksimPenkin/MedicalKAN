# """
# @author   Maksim Penkin
# """

import data
from utils.torch_utils import train_func

from nn.engines.base_engine import IEngine


class Trainer(IEngine):

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataloader(self):
        return self._eval_dataloader

    def __init__(self, db, val_db=None, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

        self._train_dataloader = data.get(db)
        if val_db is not None:
            self._eval_dataloader = data.get(val_db)
        else:
            self._eval_dataloader = None

    def train(self, *args, **kwargs):
        train_func(self.model, self.train_dataloader, *args, val_dataloader=self.eval_dataloader, **kwargs)
