# """
# @author   Maksim Penkin
# """

from lightning import Trainer
from ..nn import callbacks as cbs
from ..nn import loggers as lgs


class CommonLitTrainer(Trainer):

    def __init__(self, *args, callbacks=None, logger=None, **kwargs):
        callbacks = [cbs.get(c) for c in callbacks]
        logger = lgs.get(logger)
        super(CommonLitTrainer, self).__init__(*args, callbacks=callbacks, logger=logger, **kwargs)
