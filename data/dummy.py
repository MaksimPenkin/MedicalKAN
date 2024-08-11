# """
# @author   Maksim Penkin
# """

from data.datasets.dummy_dataset import RandomUniformDataset
from torch.utils.data import DataLoader


def random_uniform(shapes, dtypes=None, n=1, **kwargs):
    db = RandomUniformDataset(*shapes, dtypes=dtypes, n=n)
    return DataLoader(db, **kwargs)
