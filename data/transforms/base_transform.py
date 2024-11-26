# """
# @author   Maksim Penkin
# """

import abc


class ITransform:

    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, *args):
        raise NotImplementedError("Must be implemented in subclasses.")
