import os
import time
import numpy as np

from config.serving_config import topk_fun, Baseconfig
from model.model_init import DisModel
from lib.entry import EntryMetaData


def predict(EntryMetaData, config=Baseconfig):
    inputs_ids = []
    for item in EntryMetaData.entry_data:
        inputs_ids.append(item.get_all_tokens())
    if not EntryMetaData:
        raise RuntimeError(f"The input of model can not be None.")
    if not config:
        raise RuntimeError(f"The config of model can not be None.")

    is_inc_infer = not EntryMetaData.is_prompt
    if EntryMetaData.sampling_params:
        [top_p, topk_num] = EntryMetaData.sampling_params
    else:
        top_p = Baseconfig.top_p
        topk_num = Baseconfig.top_k_num

    frequency_penalty = config.frequency_penalty
    presence_penalty = config.presence_penalty
    print(f"----------------------------- begin---------", flush=True)
    time_start = time.time()
    outputs = predict_stage(inputs_ids, is_inc_infer, top_p, topk_num, frequency_penalty, presence_penalty)

    print(f"time cost {(time.time() - time_start) * 1000}ms, request  get reply. ",
          flush=True)

    return outputs


def predict_stage(input_ids, is_inc_infer, top_p, top_k_num, frequency_penalty, presence_penalty):
    """
    predict
    """
    outputs = []
    # Init outputs with original inputs
    origin_inputs = np.array([input_ids])
    _, valid_length = origin_inputs.shape
    # If target length exceeds seq_length, use seq_length instead
    vocab_size = Baseconfig.vocab_size
    seq_length = Baseconfig.seq_length
    # A list of the frequency of each token
    frequency_list = np.array([[0 for _ in range(vocab_size)]])
    pad_length = seq_length - origin_inputs.shape[-1]
    # Pad original inputs to seq_length
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))

    # Indicate the exact token position
    current_index = valid_length - 1 if valid_length - 1 > 0 else 0
    current_index = np.array([current_index], np.int32)
    batch_valid_length = np.array([current_index], np.int32)
    # For first graph, not_init should be false
    init_true = True
    init_false = False
    init = init_true and is_inc_infer
    # Call a single inference with input size of (bs, seq_length)
    logits = DisModel.call(np.array(input_ids, np.int32),
                                      current_index, batch_valid_length, init)

    # Reshape the output logits
    log_probs = logits.reshape(1, vocab_size)
    # Get the revised log_probs considering frequency and presence penalty to eliminate duplicate in generated results
    log_probs = log_probs.reshape(1, vocab_size)
    log_probs_revised = log_probs - frequency_list * frequency_penalty - (frequency_list > 0) * presence_penalty
    # Convert the log_probs to probability
    logits = np.power(10, np.array(log_probs_revised, np.float32))

    # sampling
    if top_p < 1.0:
        # Only consider the 5000 largest logits to reduce computation
        sorted_logits, index = topk_fun(logits, 5000)
        cumsum_logits = np.cumsum(sorted_logits, 1)
        cumsum_logits = cumsum_logits[0]
        index = index[0]
        sorted_logits = sorted_logits[0]
        top_p_num = sum(cumsum_logits > top_p)
        # In case the probability is smooth, the sum of 5000 largest probabilities are not large enough
        if top_p_num == 0:
            top_p_num = 5000
        # Get the corresponding probs and indices
        probs = sorted_logits[:top_p_num]
        p_args = index[:top_p_num]
        p = probs / sum(probs)
    # if top_p is set to 1.0, use top_k sampling
    else:
        # Get the corresponding probs and indices
        probs, p_args = topk_fun(logits, top_k_num)
        probs = probs[0]
        p_args = p_args[0]
        # Avoid rounding error
        if sum(probs) == 0:
            probs = np.array([1 / top_k_num for _ in range(top_k_num)])
        p = probs / sum(probs)
        # Random select a token as final output for this round
    target_index = np.random.choice(len(p), p=p)
    target = p_args[target_index]
    frequency_list[0][target] = frequency_list[0][target] + 1
    outputs.append(int(target))

    return outputs
