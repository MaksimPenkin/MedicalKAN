# """
# @author   Maksim Penkin
# """

from lightning import LightningModule
from .base_model import CommonLitModel
from .mri_enhancement_model import MRIEnhancementModel
from .ultrasound_segmentation_model import UltrasoundSegmentationModel

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(identifier,
                        module_objects={
                            "CommonLitModel": CommonLitModel,
                            "MRIEnhancementModel": MRIEnhancementModel,
                            "UltrasoundSegmentationModel": UltrasoundSegmentationModel},
                        **kwargs)

    if isinstance(obj, LightningModule):
        return obj
    raise ValueError(f"Could not interpret model instance: {obj}.")
