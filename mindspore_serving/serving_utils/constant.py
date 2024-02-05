PADDING_REQUEST_ID = "padding_request_id"
PADDING_PROMPT = "paddind prompt"

RETURN_REASON_PREDICT_FAILED = "error304: generation task stop abnormally!, please retry!"

INPUT_OUT_OF_TOKEN = (-202, 0.0)
INPUT_EMPTY_TOKEN = (-203, 0.0)
PREDICT_FAILED_CODE = -1

INTERNLM_PROMPT_FORMAT = "<s><s><|User|>:{}<eoh>\n<|Bot|>:"
BAICHUAN_PROMPT_FORMAT = "<reserved_106>{}<reserved_107>"