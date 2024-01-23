import importlib


class Registry:
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def get_obj_map(self):
        return self._obj_map

    def get_name(self):
        return self._name

    def _do_register(self, name, obj, suffix=None):
        if isinstance(suffix, str):
            name = name + '_' + suffix
        # logging.debug(f'name is  {name}.')
        # logging.debug(f'obj is  {obj}')
        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = obj

    def register(self, obj=None, suffix=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class, suffix)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj, suffix)

    def get(self, name, suffix='basicsr'):
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + '_' + suffix)
            print(f'Name {name} is not found, use name: {name}_{suffix}!')
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()

    def get_instance_from_cfg(self, cfg):
        if not isinstance(cfg, dict):
            raise TypeError("Cfg must be a Config, but got {}".format(type(cfg)))
        if 'type' not in cfg:
            raise KeyError('`cfg` must contain the key "type",' 'but got {}\n'.format(cfg))

        args = cfg.copy()
        obj_type = args.pop('type')
        obj_cls = self._obj_map.get(obj_type)
        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)('{}: {}'.format(obj_cls.__name__, e))


class Registers:
    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    MODEL_INTERFACE = Registry('model_interface')
    TOKENIZER = Registry('tokenizer')
    BASIC_INPUTS = Registry('basic_inputs')
    EXTRA_INPUTS = Registry('extra_inputs')
    WARMUP_INPUTS = Registry('warmup_inputs')




config = {
    'class_name': 'Model2',
    'params': {},
}


def _handle_errors(errors):
    if not errors:
        return
    for name, err in errors:
        print("Module {} import failed: {}".format(name, err))
    print("Please check these modules.")


