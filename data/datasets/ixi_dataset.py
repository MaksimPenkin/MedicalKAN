# """
# @author   Maksim Penkin
# """

from data.datasets.file_dataset import FileDataset


class IXIRingingDataset(FileDataset):

    def __init__(self, *args, **kwargs):
        super(IXIRingingDataset, self).__init__(*args, **kwargs)
