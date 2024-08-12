# """
# @author   Maksim Penkin
# """

import abc, os


class ISampler:

    @property
    def data(self):
        return self._data

    def __init__(self):
        self._data = None

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

    def __next__(self):
        if (self.head < 0) or (self.head >= self.__len__() - 1):
            self._reset()
        self.head += 1

        return self[self.head]


class PathSampler(IterableSampler):

    @property
    def with_names(self):
        return self._with_names

    def __init__(self, filename, root="", with_names=False):
        super(PathSampler, self).__init__()

        self._set_data(filename)
        self._root = root
        self._with_names = bool(with_names)

    @abc.abstractmethod
    def _set_data(self, filename):
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def _get_item(self, item):
        raise NotImplementedError("Must be implemented in subclasses.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        filenames = self._get_item(item)

        sample = tuple(os.path.join(self._root, filename) for filename in filenames)
        if self.with_names:
            sample += (os.path.split(filenames[0])[-1], )

        return sample
