# """
# @author   Maksim Penkin
# """

from data import samplers
from data import transforms as augmentations

from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, root, transforms=None):
        self.root = root

        if transforms is not None:
            self.transforms = augmentations.CompositeTransform([augmentations.get(t) for t in transforms])
        else:
            self.transforms = None

    def __len__(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def __getitem__(self, idx):
        raise NotImplementedError("Must be implemented in subclasses.")


class SamplerDataset(CustomDataset):

    def __init__(self, sampler, *args, **kwargs):
        super(SamplerDataset, self).__init__(*args, **kwargs)

        self.sampler = samplers.get(sampler)

    def __len__(self):
        return len(self.sampler)
