import importlib
import os
import sys


class Registry():
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
        print(f'name is  {name}.')
        print(f'obj is  {obj}')
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


class registers:
    def __init__(self):
        raise RuntimeError("Registries is not intended to be instantiated")

    MODEL_INTERFACE = Registry('model_interface')
    TOKENIZER = Registry('tokenizer')


TOKENIZER_MODULES = ['internlm_tokenizer', 'llama_tokenizer']
ALL_MODULES = [("models.tokenizer", TOKENIZER_MODULES)]


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


def import_all_modules_for_register(custom_module_paths=None):
    errors = []
    for base_dir, modules in ALL_MODULES:
        for name in modules:
            try:
                if base_dir != "":
                    full_name = base_dir + "." + name
                else:
                    full_name = name
                print(f'>>>>>>full_name is {full_name}, type is {type(full_name)}')
                print(importlib.import_module(full_name))

            except ImportError as error:
                errors.append((name, error))

    _handle_errors(errors)
    # print(build_from_config(config, MODEL))


def build_from_config(config, REGISTRY):
    name = config['class_name']
    print(f'name in build from config {name}')
    params = config['params']
    for key, target in REGISTRY._obj_map.items():
        print(f'123 key is  {key}')
        if key == name:
            return target(**params)
    return None


if __name__ == "__main__":
    print('Registers.model.dict before: ', MODEL.get_obj_map(), MODEL.get_name())
    print(f'model dict type is {type(MODEL.get_obj_map())}')
    import_all_modules_for_register()
    print(build_from_config(config, MODEL))
    print('Registers.model.dict after: ', MODEL.get_obj_map(), MODEL.get_name())