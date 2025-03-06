# """
# @author   Maksim Penkin
# """

from datetime import datetime
from ..nn import callbacks as cbs
from ..nn import loggers as lgs

from lightning import Trainer


class CommonLitTrainer(Trainer):

    def __init__(self, *args, callbacks=None, logger=None, **kwargs):
        callbacks = [cbs.get(c) for c in callbacks]
        logger = lgs.get(logger, version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        super(CommonLitTrainer, self).__init__(*args, callbacks=callbacks, logger=logger, **kwargs)
