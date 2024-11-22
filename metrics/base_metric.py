# """
# @author   Maksim Penkin
# """

import abc


class IMetric:

    @property
    def name(self):
        return self._name

    def __init__(self, name=None):
        name = name or "metric"
        assert isinstance(name, str)
        self._name = name

    @abc.abstractmethod
    def reset_state(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def update_state(self, value, n=1):
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def result(self):
        raise NotImplementedError("Must be implemented in subclasses.")
