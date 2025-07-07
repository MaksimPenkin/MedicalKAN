# """
# @author   Maksim Penkin
# """

from pathlib import Path
from src.utils.serialization_utils import create_func

from .sampler_dataset import SamplerDataset


class FileDataset(SamplerDataset):
    @property
    def root(self):
        return self._root

    def __init__(self, root, load_func, load_params, *args, **kwargs):
        super(FileDataset, self).__init__(*args, **kwargs)

        self._root = Path(root)
        self._load_func = create_func(load_func)
        self._load_params = load_params

    def _load(self, x):
        if isinstance(x, dict):
            return {k: self._load_func(self.root / filename, **kwargs) for (k, filename), kwargs in zip(x.items(), self._load_params)}
        elif isinstance(x, (list, tuple)):
            return [self._load_func(self.root / filename, **kwargs) for filename, kwargs in zip(x, self._load_params)]
        else:
            return self._load_func(self.root / x, **self._load_params)

    def __getitem__(self, index):
        sample = self._load(self.sampler[index])
        if self.transform:
            sample = self.apply_transform(sample)
        return sample
