# """
# @author   Maksim Penkin
# """

from data.datasets.image_dataset import ImageDataset


class IXIRingingDataset(ImageDataset):

    def __init__(self, *args, **kwargs):
        super(IXIRingingDataset, self).__init__(*args, **kwargs)
