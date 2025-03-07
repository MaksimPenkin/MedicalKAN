# """
# @author   Maksim Penkin
# """

import torch


TORCH_DTYPES = {
    "float32": torch.float32,
    "float": torch.float,
    "float64": torch.float64,
    "double": torch.double,
    "complex64": torch.complex64,
    "cfloat": torch.cfloat,
    "complex128": torch.complex128,
    "cdouble": torch.cdouble,
    "float16": torch.float16,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.short,
    "int32": torch.int32,
    "int": torch.int,
    "int64": torch.int64,
    "long": torch.long,
    "bool": torch.bool,
}


def torch_device():
    if torch.cuda.is_available():
        return {
            "cuda": True,
            "device_count": torch.cuda.device_count(),
            "device_current": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
        }
    else:
        return {"cuda": False}


def torch_dtype(dtype):
    if dtype is None:
        return None
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        return TORCH_DTYPES[dtype]
    else:
        raise TypeError(f"Expected `dtype` to be None, torch.dtype or str, found: {dtype} of type {type(dtype)}.")


def torch_random(size, low=0, high=None, dtype=None, device="cpu"):
    dtype = torch_dtype(dtype)

    if dtype in (torch.bool,):
        return torch.randint(0, 2, size, dtype=dtype, device=device)  # The `high` value is hard-coded.
    elif dtype in (torch.uint8, torch.int8, torch.int16, torch.short, torch.int32, torch.int, torch.int64, torch.long):
        return torch.randint(low, high, size, dtype=dtype, device=device)  # The `high` value is hard-coded.
    else:
        high = high or 1
        assert low < high, f"random_ expects `from` to be less than `to`, but found: {low} >= {high}."
        return (low - high) * torch.rand(size, dtype=dtype, device=device) + high


def torch_load(model, ckpt, **kwargs):
    checkpoint = torch.load(ckpt, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
        state_dict = {name[6:]: checkpoint[name] for name in checkpoint}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, **kwargs)
    return model


######################################################################################################################################################


def pythonify_logs(logs):
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in logs.items()}


def split_loss_logs(value):
    if isinstance(value, dict):
        return value["loss"], pythonify_logs(value)
    elif isinstance(value, (list, tuple)):
        loss, logs = value
        return loss, pythonify_logs(logs)
    else:
        return value, {"loss": value.item()}


def to_device(obj, device="cpu"):
    if isinstance(obj, torch.Tensor) or isinstance(obj, torch.nn.Module):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: to_device(v, device=device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_device(v, device=device) for v in obj]
    else:
        raise TypeError(f"Expected `obj` to be dict, list or tuple, torch.Tensor or torch.nn.Module, found: {obj} of type {type(obj)}.")
