

import numpy as np
TOPP_NUM = 1


def topk(x, top_k, axis=-1, largest=True, sort=True):
    """numpy implemented topk sample."""
    # safety check
    if x.shape[axis] < top_k:
        top_k = x.shape[axis] - 1
    if largest:
        topk_index = np.argpartition(-x, top_k, axis=axis)
    else:
        topk_index = np.argpartition(x, top_k, axis=axis)
    topk_index = np.take(topk_index, np.arange(top_k), axis=axis)
    topk_data = np.take_along_axis(x, topk_index, axis=axis)
    if sort:
        sort_index = (
            np.argsort(-topk_data, axis=axis)
            if largest
            else np.argsort(topk_data, axis=axis)
        )
        topk_data = np.take_along_axis(topk_data, sort_index, axis=axis)
        topk_index = np.take_along_axis(topk_index, sort_index, axis=axis)
    return topk_data, topk_index


def softmax_np(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


import numpy as np


def softmax_matrix(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x


def post_sampling(logits, decode_params, targets, origin_index):

    top_p = decode_params.top_p
    top_k_num = decode_params.top_k
    sorted_logits, index = topk(logits, top_k_num)
    if top_p < 1:
        cumsum_logits = softmax_np(sorted_logits)
        cumsum_logits = np.cumsum(cumsum_logits, axis=-1)
        top_p_num = sum(cumsum_logits < top_p)
            # In case the probability is smooth, the sum of 5000 largest probabilities are not large enough
        if top_p_num == 0:
            top_p_num = TOPP_NUM
            # Get the corresponding probs and indices
        sorted_logits = sorted_logits[:top_p_num]
        index = index[:top_p_num]

    p_args = index
    p = softmax_np(sorted_logits)
    target_index = np.random.choice(len(p), p=p)
    targets[origin_index] = p_args[target_index]