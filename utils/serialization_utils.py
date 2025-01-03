# """
# @author   Maksim Penkin
# """

import os, json, yaml
from importlib import import_module


def load_config(filename):
    ext = os.path.splitext(filename)[-1].lower()

    if ext in (".json", ):
        with open(filename, "r") as f:
            config = json.load(f)
    elif ext in (".yaml", ".yml"):
        with open(filename, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Expected `.json`, `.yaml` or `.yml` file, found: {filename}.")

    return config


def create_config(identifier, **kwargs):
    """
    Configuration file convention:
    {
        "class_name": ... ,
        "config": {
            ...
        }
    }
    """

    if identifier is None:
        return None

    if isinstance(identifier, str):
        if os.path.isfile(identifier):
            config = load_config(identifier)
        else:
            config = {"class_name": str(identifier), "config": {}}
    elif isinstance(identifier, dict):
        config = identifier
    else:
        raise TypeError(f"Expected `identifier` to be None, str or dict, found: {identifier} of type {type(identifier)}.")

    assert ("class_name" in config) and ("config" in config), f"Configuration file structure error: `class_name` or `config` is not found: {config}."
    config["config"].update(kwargs)

    return config


def create_object_from_config(config, module_objects=None):
    """
    Usage example,

    from nn.models.resnet import resnet18

    obj = create_object_from_config(config, module_objects={"resnet18": resnet18})
    """

    # TODO: implement registration decorator @saving.register_serializable(package='my_package')
    #       and merge that register with `module_objects` (optional look-up table, provided by user)

    if config is None:
        return None

    cls_module, cls_name, cls_config = config.get("module", None), config["class_name"], config["config"]
    if not cls_module:
        cls = module_objects[cls_name]  # If module name is not provided by the config, try to find the target class in table.
    else:
        cls = getattr(import_module(cls_module), cls_name)  # Otherwise, dynamically import the target class.

    return cls(**cls_config)  # Construct and return target object.


def create_object(identifier, module_objects=None, **kwargs):
    if identifier is None:
        return None

    try:
        config = create_config(identifier, **kwargs)
    except TypeError:
        obj = identifier
    else:
        obj = create_object_from_config(config, module_objects=module_objects)

    return obj


def create_func(identifier):
    if identifier is None:
        return None

    if isinstance(identifier, str):
        func_module, func_name = '.'.join(identifier.split('.')[:-1]), identifier.split('.')[-1]
        func = getattr(import_module(func_module), func_name)
    elif callable(identifier):
        func = identifier
    else:
        raise TypeError(f"Expected `identifier` to be None, str or callable, found: {identifier} of type {type(identifier)}.")

    return func
