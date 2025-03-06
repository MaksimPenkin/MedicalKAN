# """
# @author   Maksim Penkin
# """

from ..nn import callbacks as cbs
from ..nn import loggers as lgs

from lightning import Trainer


class CommonLitTrainer(Trainer):

    def __init__(self, *args, callbacks=None, logger=None, **kwargs):
        callbacks = [cbs.get(c) for c in callbacks]
        logger = lgs.get(logger)
        super(CommonLitTrainer, self).__init__(*args, callbacks=callbacks, logger=logger, **kwargs)
