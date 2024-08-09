# """
# @author   Maksim Penkin
# """

import abc


class ISampler:

    @property
    def multiplicity(self):
        return 1

    def __init__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def __iter__(self):
        return self

    def __call__(self):
        return self.__iter__()


class IterableSampler(ISampler):

    def __init__(self):
        super(IterableSampler, self).__init__()

        self.head = -1

    def _reset(self):
        self.head = -1

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError("data/samplers/base_sampler.py: class IterableSampler: def __getitem__(...): "
                            f"error: expected `item` be an integer instance, found: {item} of type {type(item)}.")
        if (item < 0) or (item >= self.__len__()):
            raise IndexError("data/samplers/base_sampler.py: class IterableSampler: def __getitem__(...): "
                             f"error: invalid `item` value (out of range) found: {item}.")

    def __next__(self):
        if (self.head < 0) or (self.head >= self.__len__() - 1):
            self._reset()
        self.head += 1

        return self[self.head]
