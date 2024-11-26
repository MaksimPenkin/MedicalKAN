# """
# @author   Maksim Penkin
# """

import abc


class ISampler:

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

    def __next__(self):
        # Reset sampler, if head is on the limits.
        if (self.head < 0) or (self.head >= self.__len__() - 1):
            self.head = -1
        # Locate head on the current element.
        self.head += 1
        # Return the current element.
        return self[self.head]
