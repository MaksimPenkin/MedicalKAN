# """
# @author   Maksim Penkin
# """

from data.datasets.image_dataset import ImageDataset
from data.samplers.csv_sampler import CSVSampler
from torchvision import transforms

from torch.utils.data import DataLoader


def ixi(filename, root="", keys=("sketch", "gt"), **kwargs):
    db = ImageDataset(CSVSampler(filename, root=root),
                      keys=keys,
                      transform=transforms.Compose([transforms.ToTensor()]))

    return DataLoader(db, **kwargs)
