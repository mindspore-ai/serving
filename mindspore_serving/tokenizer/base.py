class BaseTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    """
    BaseInputsOfInfer interface.
    """

    def encode(self, input_prompts):
        pass

    def decode(self, input_token):
        pass