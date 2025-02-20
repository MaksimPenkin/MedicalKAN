# """
# @author   Maksim Penkin
# """

import torch
from nn import models


class IEngine:

    @property
    def model(self):
        return self._model

    def __init__(self, model, checkpoint=None):
        self._model = models.get(model, checkpoint=checkpoint)

    @staticmethod
    def _to_ckpt(model, save_path=None):
        model.to("cpu")
        model.eval()
        with torch.no_grad():
            torch.save(model.state_dict(), save_path or "model.pt")

    @staticmethod
    def _to_onnx(model, *args, save_path=None, **kwargs):
        # TODO: Extending the ONNX Registry: https://pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html
        """
        torch.onnx.export(model,              # model being run
                          x,                  # model input (or a tuple for multiple inputs)
                          "model.onnx",       # where to save the model (can be a file or file-like object)
                          opset_version=14    # the ONNX version to export the model to
                          input_names=None,   # the model's input names
                          output_names=None,  # the model's output names
                          dynamic_axes=None)  # variable length axes
        """
        model.to("cpu")
        model.eval()
        with torch.no_grad():
            torch.onnx.export(model, args, save_path or "model.onnx", **kwargs)

    @staticmethod
    def _to_jit(model, *args, save_path=None, **kwargs):
        """
        torch.jit.trace(model,  # A Python function or torch.nn.Module that will be run with example_inputs.
                        args,   # (tuple or torch.Tensor or None) – A tuple of example inputs that will be passed to the function while tracing.
                        ...)
        """
        model.to("cpu")
        model.eval()
        with torch.no_grad():
            jit_model = torch.jit.trace(model, args, **kwargs)
            torch.jit.save(jit_model, save_path or "model.jit.pt")

    @classmethod
    def export(cls, model, shapes, dtypes=None, option=None, save_path=None, **kwargs):
        from data.dummy import random_uniform

        example_inputs = next(iter(random_uniform(shapes, dtypes=dtypes)))

        if option is None:
            cls._to_ckpt(model, save_path=save_path)
        elif option == "onnx":
            cls._to_onnx(model, *example_inputs, save_path=save_path, **kwargs)
        elif option == "jit":
            cls._to_jit(model, *example_inputs, save_path=save_path, **kwargs)
        else:
            raise ValueError(f"Unrecognized `option` found: {option}. "
                             "Only None, `onnx` or `jit` are supported.")
