import logging
from mindspore_serving.config.config import ServingConfig
from mindspore_serving.serving_utils.register import Registers
from mindformers.mindformer_book import MindFormerBook
from mindformers import AutoTokenizer
from transformers import AutoTokenizer as TransfomersTokenizer
from transformers import LlamaTokenizer
from research.baichuan2.baichuan2_tokenizer import Baichuan2Tokenizer


def build_tokenizer(base_config: ServingConfig = None):
    if base_config is None:
        return None
    if base_config is not None:
        if isinstance(base_config, dict) and not isinstance(base_config, ServingConfig):
            base_config = ServingConfig(base_config)
        # TODO
        # check_and_add_vocab_file_path(base_config)
        # return Registers.TOKENIZER.get_instance_from_cfg(base_config)
    tokenizer_type = base_config.tokenizer.type
    logging.info(f'tokenizer tokens is {base_config.tokenizer}')
    if tokenizer_type in MindFormerBook.get_tokenizer_support_list():
        logging.info('load tokenizer from mindformers')
        tokenizer = AutoTokenizer.from_pretrained(base_config.tokenizer.type)
    elif tokenizer_type == 'LlamaTokenizer':
        # tokenizer = LlamaTokenizer(base_config.tokenizer_path)
        tokenizer = LlamaTokenizer(base_config.tokenizer.vocab_file)
        logging.info(f'tokenizer special tokens is {tokenizer.all_special_tokens}')
        return tokenizer
        # logging.info('load custom tokenizer')

    # 加入百川tokenlizer
    elif tokenizer_type == 'BaichuanTokenizer':
        tokenizer = Baichuan2Tokenizer(base_config.tokenizer.vocab_file)
    elif tokenizer_type == 'WizardCoderTokenizer':
        tokenizer = TransfomersTokenizer.from_pretrained(base_config.tokenizer.vocab_file)
    else:
        tokenizer = Registers.TOKENIZER.get_obj_map()[tokenizer_type](base_config.tokenizer.vocab_file)
    return tokenizer
