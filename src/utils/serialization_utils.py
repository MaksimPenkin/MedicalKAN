# """
# @author   Maksim Penkin
# """

import os, re, yaml, functools
from pathlib import Path
from importlib import import_module


class ConfigLoaderMeta(type):
    def __new__(metacls, name, bases, namespace):
        cls = super().__new__(metacls, name, bases, namespace)
        # 1. Add !include tag processing.
        cls.add_constructor("!include", cls.construct_include)
        # 2.1 Add !path tag processing.
        cls.add_constructor("!path", cls.construct_path)
        # 2.2 Add !path tag matcher.
        cls._path_matcher = re.compile(r"\$\{([^}^{]+)\}")  # re.compile('.*?\${(\w+)}.*?')
        cls.add_implicit_resolver("!path", cls._path_matcher, None)  # Note: An implicit resolver can only match plain scalars, not quoted.

        return cls


class ConfigLoader(yaml.Loader, metaclass=ConfigLoaderMeta):
    def __init__(self, stream):
        try:
            self._root = Path(stream.name).parent
        except AttributeError:
            self._root = Path()

        super().__init__(stream)

    def construct_include(self, node):
        filename = self._root / self.construct_scalar(node)
        ext = filename.suffix.lower()

        with open(filename, "r") as f:
            if ext in (".yaml", ".yml", ".json"):  # YAML can load JSON.
                return yaml.load(f, ConfigLoader)
            else:
                return "\n".join(ln for ln in (line.strip() for line in f.read().splitlines()) if ln)

    def construct_path(self, node):
        # match():   Determine if the RE matches at the beginning of the string.
        # group():   Return the string matched by the RE.
        # findall(): Find all substrings where the RE matches, and returns them as a list.

        return re.sub(self._path_matcher, lambda m: os.environ.get(m.group()[2:-1]), self.construct_scalar(node))


def load_config(filename):
    load = functools.partial(yaml.load, Loader=ConfigLoader)
    with open(filename, "r") as f:
        config = load(f)

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

    if isinstance(identifier, os.PathLike):
        identifier = os.fspath(identifier)
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


def create_object_from_config(config, module_objects=None, partial=False):
    """
    Usage example,

    from nn.models.resnet import resnet18

    obj = create_object_from_config(config, module_objects={"resnet18": resnet18})
    """

    # TODO: implement registration decorator @saving.register_serializable(package='my_package')
    #       and merge that register with `module_objects` (optional look-up table, provided by user)

    if config is None:
        return None

    cls_module, cls_name, cls_config = config.get("module"), config["class_name"], config["config"]
    if not cls_module:
        cls = module_objects[cls_name]  # If module name is not provided by the config, try to find the target class in table.
    else:
        cls = getattr(import_module(cls_module), cls_name)  # Otherwise, dynamically import the target class.

    # Construct and return target object.
    if not partial:
        return cls(**cls_config)
    else:
        return functools.partial(cls, **cls_config)


def create_object(identifier, module_objects=None, partial=False, **kwargs):
    if identifier is None:
        return None

    try:
        config = create_config(identifier, **kwargs)
    except TypeError:
        obj = identifier
    else:
        obj = create_object_from_config(config, module_objects=module_objects, partial=partial)

    return obj


def create_func(identifier):
    if identifier is None:
        return None

    if isinstance(identifier, os.PathLike):
        identifier = os.fspath(identifier)
    if isinstance(identifier, str):
        func_module, func_name = ".".join(identifier.split(".")[:-1]), identifier.split(".")[-1]
        func = getattr(import_module(func_module), func_name)
    else:
        func = identifier

    if callable(func):
        return func
    else:
        raise ValueError(f"Expected callable `func`, found: {func} of type {type(func)}.")
