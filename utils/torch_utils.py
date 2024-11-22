# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

import json, time
from tqdm import tqdm
from datetime import datetime

import numpy as np
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
    "bool": torch.bool
}


def torch_device():
    if torch.cuda.is_available():
        return json.dumps({
            "cuda": True,
            "device_count": torch.cuda.device_count(),
            "device_current": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0)}, indent=4)
    else:
        return json.dumps({"cuda": False}, indent=4)


def torch_dtype(dtype):
    if dtype is None:
        return None
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        return TORCH_DTYPES[dtype]
    else:
        raise TypeError(f"Expected `dtype` to be None, torch.dtype or str, found: {dtype} of type {type(dtype)}.")


def torch_random(size, dtype=None, device="cpu"):
    dtype = torch_dtype(dtype)

    if dtype == torch.bool:
        return torch.randint(0, 2, size, dtype=dtype, device=device)  # The `high` value is hard-coded.
    elif dtype in [torch.uint8, torch.int8, torch.int16, torch.short, torch.int32, torch.int, torch.int64, torch.long]:
        return torch.randint(0, 10, size, dtype=dtype, device=device)  # The `high` value is hard-coded.
    else:
        return torch.rand(size, dtype=dtype, device=device)


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


def _pythonify_logs(logs):
    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in logs.items()}


def _split_loss_logs(value):
    if isinstance(value, dict):
        return value["loss"], _pythonify_logs(value)
    elif isinstance(value, (list, tuple)):
        loss, logs = value
        return loss, _pythonify_logs(logs)
    else:
        return value, {"loss": value.item()}


######################################################################################################################################################


def _to_device(obj, device="cpu"):
    if isinstance(obj, torch.Tensor) or isinstance(obj, torch.nn.Module):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: _to_device(v, device=device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_device(v, device=device) for v in obj]
    else:
        raise TypeError(f"Expected `obj` to be dict, list or tuple, torch.Tensor or torch.nn.Module, found: {obj} of type {type(obj)}.")


######################################################################################################################################################


def _model_step(model, x):
    if isinstance(x, dict):
        try:
            y_pred = model(**x)
        except:
            y_pred = model(x)
    elif isinstance(x, (list, tuple)):
        try:
            y_pred = model(*x)
        except:
            y_pred = model(x)
    else:
        y_pred = model(x)

    return y_pred


def _criterion_step(criterion, y_pred, y):
    if isinstance(y, dict):
        try:
            value = criterion(y_pred, **y)
        except:
            value = criterion(y_pred, y)
    elif isinstance(y, (list, tuple)):
        try:
            value = criterion(y_pred, *y)
        except:
            value = criterion(y_pred, y)
    else:
        value = criterion(y_pred, y)

    return _split_loss_logs(value)


######################################################################################################################################################


def _eval_step(model, x, y, criterion, device="cpu"):
    x = _to_device(x, device=device)
    y = _to_device(y, device=device)

    with torch.no_grad():
        y_pred = _model_step(model, x)
        _, logs = _criterion_step(criterion, y_pred, y)
    return {"val_" + k: v for k, v in logs.items()}


def _train_step(model, x, y, criterion, optimizer, device="cpu"):
    x = _to_device(x, device=device)
    y = _to_device(y, device=device)

    optimizer.zero_grad()
    y_pred = _model_step(model, x)
    loss, logs = _criterion_step(criterion, y_pred, y)
    loss.backward()
    optimizer.step()
    return logs


def _inference_step(model, x, device="cpu"):
    x = _to_device(x, device=device)

    with torch.no_grad():
        y_pred = _model_step(model, x)
    return y_pred


######################################################################################################################################################


def eval_func(model, dataloader, criterion, callbacks=None, limit_batches=1.0, device="cpu"):
    from tools.optimization import losses
    from tools.optimization.callbacks import CompositeCallback
    from tools.optimization.metrics import CompositeMetric

    steps = int(limit_batches * len(dataloader))

    criterion = losses.get(criterion)
    if not isinstance(callbacks, CompositeCallback):
        callbacks = CompositeCallback(callbacks=callbacks, model=model)

    tracker = CompositeMetric()

    model.to(device)
    model.eval()
    callbacks.on_test_begin()
    eval_logs = {}
    for idx, (x, y) in enumerate(tqdm(dataloader, total=steps)):
        if idx >= steps:
            break
        # x, y = (blob["left_image"], blob["sparse_disparity"]), blob["disparity"]
        callbacks.on_test_batch_begin(idx)
        logs = _eval_step(model, x, y, criterion, device=device)
        callbacks.on_test_batch_end(idx, logs=logs)
        tracker.update_state(logs, n=x.size(0))  # TODO: add seamless batch_size value extraction.
    eval_logs = tracker.result()
    callbacks.on_test_end(logs=eval_logs)

    return eval_logs


def train_func(model, dataloader, criterion, optimizer="adam", callbacks=None, epochs=1, val_dataloader=None, limit_batches=1.0, device="cpu"):
    from tools.optimization import losses, optimizers
    from tools.optimization.callbacks import CompositeCallback
    from tools.optimization.metrics import CompositeMetric

    steps = int(limit_batches * len(dataloader))
    val_steps = len(val_dataloader) if val_dataloader is not None else None

    criterion = losses.get(criterion)
    optimizer = optimizers.get(optimizer, params=model.parameters())
    if not isinstance(callbacks, CompositeCallback):
        callbacks = CompositeCallback(callbacks=callbacks, model=model)

    train_tracker = CompositeMetric()
    val_tracker = CompositeMetric() if val_dataloader is not None else None

    model.to(device)
    callbacks.on_train_begin()
    epoch_logs = {}
    for epoch in tqdm(range(epochs)):
        train_tracker.reset_state()
        model.train()
        callbacks.on_epoch_begin(epoch)
        for idx, (x, y) in enumerate(tqdm(dataloader, total=steps, leave=False)):
            if idx >= steps:
                break
            # x, y = (blob["left_image"], blob["sparse_disparity"]), blob["disparity"]
            callbacks.on_train_batch_begin(idx)
            logs = _train_step(model, x, y, criterion, optimizer, device=device)
            callbacks.on_train_batch_end(idx, logs=logs)
            train_tracker.update_state(logs, n=x.size(0))  # TODO: add seamless batch_size value extraction.
        epoch_logs = train_tracker.result()

        if val_dataloader is not None:
            val_tracker.reset_state()
            model.eval()
            callbacks.on_test_begin()
            for idx, (x, y) in enumerate(tqdm(val_dataloader, total=val_steps, leave=False)):
                if idx >= val_steps:
                    break
                # x, y = (blob["left_image"], blob["sparse_disparity"]), blob["disparity"]
                callbacks.on_test_batch_begin(idx)
                logs = _eval_step(model, x, y, criterion, device=device)
                callbacks.on_test_batch_end(idx, logs=logs)
                val_tracker.update_state(logs, n=x.size(0))  # TODO: add seamless batch_size value extraction.
            val_logs = val_tracker.result()
            callbacks.on_test_end(logs=val_logs)
            epoch_logs.update(val_logs)

        callbacks.on_epoch_end(epoch, logs=epoch_logs)
    callbacks.on_train_end(logs=epoch_logs)


def inference_func(model, dataloader, limit_batches=1.0, device="cpu"):
    steps = int(limit_batches * len(dataloader))
    model.to(device)
    model.eval()

    try:
        for idx, (x, y) in enumerate(tqdm(dataloader, total=steps)):
            if idx >= steps:
                break
            _ = _inference_step(model, x, device=device)
    except:
        for idx, x in enumerate(tqdm(dataloader, total=steps)):
            if idx >= steps:
                break
            _ = _inference_step(model, x, device=device)


def latency_func(model, shapes, dtypes=None, warmup=10, iteration=100, device="cpu"):
    from tools.optimization.datasets.dummy import random_uniform

    model.to(device)
    model.eval()
    example_inputs = _to_device(next(iter(random_uniform(shapes, dtypes=dtypes))), device=device)

    print(f"{datetime.now()}: Warm up...")
    with torch.no_grad():
        for i in range(warmup):
            _ = model(*example_inputs)
    torch.cuda.synchronize()

    print(f"{datetime.now()}: Start timing...")
    timings = []
    with torch.no_grad():
        for i in range(iteration):
            start_time = time.time()
            _ = model(*example_inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append((end_time - start_time) * 1000)

    print(f"{datetime.now()}: "
          "Latency:\n"
          f"    min = {np.amin(timings):.5f} ms\n"
          f"    max = {np.amax(timings):.5f} ms\n"
          f"    mean = {np.mean(timings):.5f} ms\n"
          f"    median = {np.median(timings):.5f} ms\n"
          f"    percentile(90 %) = {np.percentile(timings, 90):.5f} ms\n"
          f"    percentile(95 %) = {np.percentile(timings, 95):.5f} ms\n"
          f"    percentile(99 %) = {np.percentile(timings, 99):.5f} ms")
