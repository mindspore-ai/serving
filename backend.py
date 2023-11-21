import abc
import  mindformers as mf

class Backend(metaclass=abc.ABCMeta):

    def __init__(self, tokenizer, modelname, model0_path, model1_path, seq_length=None):
        self.tokenizer = tokenizer
        self.modelname = modelname
        self.model0_path = model0_path
        self.model1_path = model1_path
        self.prefill_model = mf.load(self.model0_path)
        self.increment_model = mf.load(self.model1_path)
        self.seq_length = seq_length

    @staticmethod
    def _pre_processing(modelname, prompt_token_ids, is_prefill=True):
        # 模型差异化前处理和打包输入
        wrapped_inputs = mf.pre_precessing(modelname, prompt_token_ids, is_prefill)
        return wrapped_inputs

    def _get_inputs(self, prompt_token_ids, is_prefill=True):
        # 输入打包
        if is_prefill:
            wrappered_inputs = self._pre_processing(self.modelname, prompt_token_ids, is_prefill, self.seq_length)
        else:
            wrappered_inputs = self._pre_processing(self.modelname, prompt_token_ids, is_prefill, 1)
        return wrappered_inputs

    @abc.abstractmethod
    def pre_process(self, prompt, is_prefill):

        prompt_token_ids = self.tokenizer(prompt)

        return self._get_inputs(prompt_token_ids, is_prefill=is_prefill)

    @abc.abstractmethod
    def predict(self, input_list=None, is_prefill=True):
        if is_prefill:
            return self.prefill_model.predict(input_list)
        else:
            return self.increment_model.predict(input_list)

    @abc.abstractmethod
    def post_process(self, output_list, temperature=1.0, top_k=1, top_p=1.0, do_sample=True):
        # 后处理（入图）
        return mf.post_process(self.modelname, output_list, temperature, top_k, top_p, do_sample)




