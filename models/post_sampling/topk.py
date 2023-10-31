
import numpy as np

from config.serving_config import TOPP_NUM


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
    print('topk_data: ', topk_data)
    print('topk_index: ', topk_index)
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


def post_sampling(logits, top_p, top_k_num):
    # sampling
    P = None
    p_args = None
    if top_p < 1.0:
        # Only consider the 5000 largest logits to reduce computation
        sorted_logits, index = topk(logits, TOPP_NUM)
        cumsum_logits = np.cumsum(sorted_logits, 0)
        cumsum_logits = cumsum_logits
        sorted_logits = sorted_logits
        top_p_num = sum(cumsum_logits > top_p)
        # In case the probability is smooth, the sum of 5000 largest probabilities are not large enough
        if top_p_num == 0:
            top_p_num = TOPP_NUM
        # Get the corresponding probs and indices
        probs = sorted_logits[:top_p_num]
        p_args = index[:top_p_num]
        if np.sum(probs) == 0:
            probs = np.array([1 / top_p_num for _ in range(top_p_num)])
        p = softmax_np(probs)
        P = p
    # if top_p is set to 1.0, use top_k sampling
    else:
        probs, index = topk(logits, top_k_num)
        probs = probs
        p_args = index
        # Avoid rounding error

        if np.sum(probs) == 0:
            probs = np.array([1 / top_k_num for _ in range(top_k_num)])

        p = softmax_np(probs)
        P = p

    target_index = np.random.choice(len(P), p=P)
    print('target_index: ', target_index)
    target = p_args[target_index]
    return target


if __name__ == '__main__':
      a = np.array([1.2, 3, -1.5, 12, 23.7, 11.0, -3])
      p = softmax_np(a)
      print(p)

      for i in range(200):
          a = np.random.rand(i + 1)
          print(a)
          p = softmax_np(a)
          print('p', p)
          target_index = np.random.choice(len(p), p=p)
          print(target_index)