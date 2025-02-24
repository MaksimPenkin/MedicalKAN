# """
# @author   Maksim Penkin
# """

import abc


class ITransform:

    @abc.abstractmethod
    def _fn(self, x):
        raise NotImplementedError("Must be implemented in subclasses.")

    def __call__(self, sample):
        if isinstance(sample, dict):
            return {k: self._fn(v) for k, v in sample.items()}
        elif isinstance(sample, (list, tuple)):
            return [self._fn(v) for v in sample]
        else:
            return self._fn(sample)
