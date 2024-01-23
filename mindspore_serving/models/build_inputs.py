from mindspore_serving.models.model_inputs.llama_inputs import LlamaBasicInputs, LlamaExtraInputs, LlamaWarmupInputs


def build_inputs(config: dict = None, module_type: str = None):
    if config is None or module_type is None:
        return None

    if module_type == 'basic_inputs':
        obj_cls = LlamaBasicInputs()
    elif module_type == 'extra_inputs':
        obj_cls = LlamaExtraInputs()
    elif module_type == 'warmup_inputs':
        obj_cls = LlamaWarmupInputs()
    else:
        raise TypeError("invalid module type, expect one of ['basic_inputs', 'extra_inputs', 'warmup_inputs'], "
                        "but got {}".format(module_type))
    return obj_cls
