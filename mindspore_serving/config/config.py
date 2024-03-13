import os
import yaml
import copy
from mindformers.tools.register.config import ordered_yaml_load


def check_valid_config(config):
    if not (config and config.model_path and config.model_config and config.serving_config and
            config.tokenizer and config.basic_inputs and config.extra_inputs and config.warmup_inputs):
        return False
    # model_path校验
    model_path = config.model_path
    if not (model_path.prefill_model and model_path.decode_model and model_path.argmax_model and model_path.topk_model
            and model_path.prefill_ini and model_path.decode_ini and model_path.post_model_ini):
        print("ERROR: there exists empty block on yaml, check the model_path part")
        return False
    if len(model_path.prefill_model) != len(model_path.decode_model):
        print("ERROR: got different size of prefill_model and decode_model")
        return False

    # model_config校验
    model_config = config.model_config
    if not (model_config.model_name and model_config.end_token is not None and model_config.vocab_size and
            model_config.prefill_batch_size and len(model_config.prefill_batch_size) > 0 and
            model_config.decode_batch_size and len(model_config.decode_batch_size) > 0):
        print("ERROR: there exists empty block on yaml, check the model_config part")
        return False
    if model_config.seq_length is None or len(model_config.seq_length) == 0:
        model_config['seq_type'] = "dyn"
    if model_config.batching_strategy is None:
        model_config['batching_strategy'] = "continuous"
    if model_config.current_index is None:
        model_config['current_index'] = False
    if model_config.page_attention is None:
        model_config['page_attention'] = False
    if model_config.backend is None:
        model_config['backend'] = 'ge'

    #   # pa_config校验
    if model_config.page_attention:   #  todo 非PA模型这里有bug
        if config.pa_config is None:
            config.pa_config ={}
            pa_config = config.pa_config
            pa_config['num_blocks'] = 224
            pa_config['block_size'] = 128
            pa_config['decode_seq_length'] = 4096

    # serving_config校验
    serving_config = config.serving_config
    if not (serving_config.agent_ports and serving_config.server_port):
        print("ERROR: there exists empty block on yaml, check the serving_config part")
        return False
    if serving_config.agent_ip is None:
        serving_config['agent_ip'] = "localhost"
    if serving_config.server_ip is None:
        serving_config['server_ip'] = "localhost"
    if serving_config.start_device_id is None:
        serving_config['start_device_id'] = 0
    if serving_config.prefill_batch_waiting_time is None:
        serving_config['prefill_batch_waiting_time'] = 0.0
    if serving_config.decode_batch_waiting_time is None:
        serving_config['decode_batch_waiting_time'] = 0.0
    if serving_config.enable_host_post_sampling is None:
        serving_config['enable_host_post_sampling'] = False
    if len(serving_config.agent_ports) != len(model_path.prefill_model):
        print("ERROR: got different size of agent_ports and models")
        return False

    return True


class ServingConfig(dict):
    """
    A Config class is inherit from dict.

    Config class can parse arguments from a config file of yaml.

    Args:
        args: yaml file name

    Example:
        test.yaml:
            a:1
        >>> cfg = ServingConfig('./test.yaml')
        >>> cfg.a
        1
    """

    def __init__(self, *args):
        super(ServingConfig, self).__init__()
        cfg_dict = {}

        # load from file
        for arg in args:
            if isinstance(arg, str):
                if arg.endswith('yaml') or arg.endswith('yml'):
                    raw_dict = ServingConfig._file2dict(arg)
                    cfg_dict.update(raw_dict)

        ServingConfig._dict2config(self, cfg_dict)

    def __getattr__(self, key):
        """Get a object attr by its `key`

        Args:
            key (str) : the name of object attr.

        Returns:
            attr of object that name is `key`
        """
        if key not in self:
            return None

        return self[key]

    def __setattr__(self, key, value):
        """Set a object value `key` with `value`

        Args:
            key (str) : The name of object attr.
            value : the `value` need to set to the target object attr.
        """
        self[key] = value

    def __delattr__(self, key):
        """Delete a object attr by its `key`.

        Args:
            key (str) : The name of object attr.
        """
        del self[key]

    def __deepcopy__(self, memo=None):
        """Deep copy operation on arbitrary MindFormerConfig objects.

        Args:
            memo (dict) : Objects that already copied.
        Returns:
            MindFormerConfig : The deep copy of the given MindFormerConfig object.
        """
        config = ServingConfig()
        for key in self.keys():
            config.__setattr__(copy.deepcopy(key, memo),
                               copy.deepcopy(self.__getattr__(key), memo))
        return config

    @staticmethod
    def _file2dict(file_name=None):
        """Convert config file to dictionary.

        Args:
            file_name (str) : config file.
        """
        if file_name is None:
            raise NameError('This {} cannot be empty.'.format(file_name))

        filepath = os.path.realpath(file_name)
        with open(filepath, encoding='utf-8') as fp:
            cfg_dict = ordered_yaml_load(fp, yaml_loader=yaml.FullLoader)
        return cfg_dict

    @staticmethod
    def _dict2config(config, dic):
        """Convert dictionary to config.

                Args:
                    config : Config object
                    dic (dict) : dictionary
                Returns:

                Exceptions:

                """
        if isinstance(dic, dict):
            for key, value in dic.items():
                if isinstance(value, dict):
                    sub_config = ServingConfig()
                    dict.__setitem__(config, key, sub_config)
                    ServingConfig._dict2config(sub_config, value)
                else:
                    config[key] = dic[key]
