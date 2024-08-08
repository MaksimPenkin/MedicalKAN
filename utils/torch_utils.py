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
        raise TypeError("tools/optimization/utils/torch_utils.py: def torch_dtype(...): "
                        f"error: expected `dtype` to be None, torch.dtype or str, found: {dtype} of type {type(dtype)}.")


def torch_random(size, dtype=None, device="cpu"):
    dtype = torch_dtype(dtype)

    if dtype == torch.bool:
        return torch.randint(0, 2, size, dtype=dtype, device=device)  # The `high` value is hard-coded.
    elif dtype in [torch.uint8, torch.int8, torch.int16, torch.short, torch.int32, torch.int, torch.int64, torch.long]:
        return torch.randint(0, 10, size, dtype=dtype, device=device)  # The `high` value is hard-coded.
    else:
        return torch.rand(size, dtype=dtype, device=device)


def move_data_device(data, device="cpu"):
    if isinstance(data, torch.Tensor):
        return data.to(device)

    if isinstance(data, dict):
        d = {k: move_data_device(v, device=device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        d = [move_data_device(v, device=device) for v in data]
    else:
        raise TypeError("tools/optimization/utils/torch_utils.py: def move_data_device(...): "
                        f"error: expected `data` to be dict, list or tuple, or torch.Tensor, found: {data} of type {type(data)}.")

    return d


def forward_wrapper(model, data, keys=None):
    # 1. Device identification.
    try:
        device = next(model.parameters()).device
    except:
        device = "cpu"

    # 2. Data transfer to the device.
    data = move_data_device(data, device=device)

    # 3. Forward.
    if isinstance(data, dict):
        if keys is not None:
            output = model(**{k: data[k] for k in keys})
        else:
            output = model(**data)
    elif isinstance(data, (list, tuple)):
        try:
            output = model(*data)
        except:
            output = model(data)
    else:
        output = model(data)

    return output


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


def _eval_step(model, x, y, criterion, keys=None):
    with torch.no_grad():
        output = forward_wrapper(model, x, keys=keys)
        _, logs = _split_loss_logs(criterion(output, y))
    return {"val_" + k: v for k, v in logs.items()}


def _train_step(model, x, y, criterion, optimizer, keys=None):
    optimizer.zero_grad()
    output = forward_wrapper(model, x, keys=keys)
    loss, logs = _split_loss_logs(criterion(output, y))
    loss.backward()
    optimizer.step()
    return logs


def train_func(model, dataloader, criterion, optimizer="adam", callbacks=None, epochs=1, val_dataloader=None,
               limit_batches=1.0, keys=None, device="cpu"):
    from nn import losses, optimizers
    from nn.callbacks.base_callback import CompositeCallback
    from metrics.base_metric import CompositeMetric

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
            y = move_data_device(y, device=device)
            callbacks.on_train_batch_begin(idx)
            logs = _train_step(model, x, y, criterion, optimizer, keys=keys)
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
                y = move_data_device(y, device=device)
                callbacks.on_test_batch_begin(idx)
                logs = _eval_step(model, x, y, criterion, keys=keys)
                callbacks.on_test_batch_end(idx, logs=logs)
                val_tracker.update_state(logs, n=x.size(0))  # TODO: add seamless batch_size value extraction.
            val_logs = val_tracker.result()
            callbacks.on_test_end(logs=val_logs)
            epoch_logs.update(val_logs)

        callbacks.on_epoch_end(epoch, logs=epoch_logs)
    callbacks.on_train_end(logs=epoch_logs)


def inference_func(model, dataloader, limit_batches=1.0, keys=None, device="cpu"):
    steps = int(limit_batches * len(dataloader))
    model.to(device)
    model.eval()

    try:
        for idx, (x, y) in enumerate(tqdm(dataloader, total=steps)):
            if idx >= steps:
                break
            with torch.no_grad():
                _ = forward_wrapper(model, x, keys=keys)
    except:
        for idx, x in enumerate(tqdm(dataloader, total=steps)):
            if idx >= steps:
                break
            with torch.no_grad():
                _ = forward_wrapper(model, x, keys=keys)


def latency_func(model, shapes, dtypes=None, warmup=10, iteration=100, device="cpu"):
    from datasets.dummy import random_uniform

    model.to(device)
    model.eval()
    example_inputs = move_data_device(next(iter(random_uniform(shapes, dtypes=dtypes))), device=device)

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
