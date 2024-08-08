# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

import abc


class IMetric:

    @property
    def name(self):
        return self._name

    def __init__(self, name=None):
        if name is None:
            self._name = "metric"
        elif isinstance(name, str):
            self._name = name
        else:
            raise ValueError("metrics/base_metric.py: class IMetric: def __init__(...): "
                             f"error: expected `name` to be string or None, found: {name} of type {type(name)}.")

    @abc.abstractmethod
    def reset_state(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def update_state(self, value, n=1):
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def result(self):
        raise NotImplementedError("Must be implemented in subclasses.")


class MeanMetric(IMetric):

    @property
    def total(self):
        return self._total

    @property
    def count(self):
        return self._count

    def __init__(self, *args, **kwargs):
        super(MeanMetric, self).__init__(*args, **kwargs)

        self._total = 0.0
        self._count = 0

        self.reset_state()

    def reset_state(self):
        self._total = 0.0
        self._count = 0

    def update_state(self, value, n=1):
        self._total += value * n
        self._count += n

    def result(self):
        if self._count == 0:
            return 0.0
        else:
            return self._total / self._count


class CompositeMetric(IMetric):

    def __init__(self, metrics=None, name=None):
        super(CompositeMetric, self).__init__(name=name)

        if metrics:
            self.metrics = {m.name: m for m in metrics}  # TODO: add isinstance(m, IMetric) check.
        else:
            self.metrics = {}

    def reset_state(self):
        for m in self.metrics.values():
            m.reset_state()

    def update_state(self, value, n=1):
        for k, v in value.items():
            if k not in self.metrics:
                self.metrics[k] = MeanMetric(name=k)
            self.metrics[k].update_state(v, n=n)

    def result(self):
        return {m.name: m.result() for m in self.metrics.values()}
