from torch.utils.data import DataLoader
from torchvision import transforms


def ixi(filename, root="", keys=("sketch", "gt"), **kwargs):
    db = ImageDataset(CSVSampler(filename, root=root),
                      keys=keys,
                      transform=transforms.Compose([transforms.ToTensor()]))

    return DataLoader(db, **kwargs)
