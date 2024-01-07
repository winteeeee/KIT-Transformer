import numpy as np


def _get_value(pos, i, d_model):
    if i & 1:
        func = np.cos
    else:
        func = np.sin

    # cos의 경우 2i + 1일 때 pos / (10000 ^ (2i / d_model))의 값이 곱해져야 하므로 i // 2를 통해 나머지를 없애줌
    return func(pos / (10000 ** ((2 * (i // 2)) / d_model)))


def positional_encode(position, d_model):
    positional_encode_vector = np.zeros((position, d_model))

    for pos, row_vector in enumerate(positional_encode_vector):
        for i, element in enumerate(row_vector):
            row_vector[i] = _get_value(pos, i, d_model)

    return positional_encode_vector
