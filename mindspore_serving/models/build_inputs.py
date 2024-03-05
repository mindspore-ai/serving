from mindspore_serving.config.config import ServingConfig
from mindspore_serving.serving_utils.register import Registers


def build_inputs(config: dict = None, module_type: str = None):
    if config is None or module_type is None:
        return None

    if isinstance(config, dict) and not isinstance(config, ServingConfig):
        config = ServingConfig(config)
    if module_type == 'basic_inputs':
        obj_cls = Registers.BASIC_INPUTS.get_instance_from_cfg(config)
    elif module_type == 'extra_inputs':
        obj_cls = Registers.EXTRA_INPUTS.get_instance_from_cfg(config)
    elif module_type == 'warmup_inputs':
        obj_cls = Registers.WARMUP_INPUTS.get_instance_from_cfg(config)
    else:
        raise TypeError("invalid module type, expect one of ['basic_inputs', 'extra_inputs', 'warmup_inputs'], "
                        "but got {}".format(module_type))
    return obj_cls
