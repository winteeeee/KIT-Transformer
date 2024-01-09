import numpy as np


def _get_value(pos, i, d_model):
    """
    pos, i, d_model이 주어졌을 때 포지셔널 인코딩 행렬의 값을 계산

    :param pos: 임베딩 행렬의 행 인덱스
    :param i: 임베딩 행렬의 열 인덱스
    :param d_model: 수식 계산에 활용될 d_model
    :return: 해당 위치의 포지셔널 인코딩 값
    """
    if i & 1:
        func = np.cos
    else:
        func = np.sin

    # cos의 경우 2i + 1일 때 pos / (10000 ^ (2i / d_model))의 값이 곱해져야 하므로 i // 2를 통해 나머지를 없애줌
    return func(pos / (10000 ** ((2 * (i // 2)) / d_model)))


def positional_encode(position, d_model):
    """
    트랜스포머에게 단어의 위치 정보를 주입하는 포지셔널 인코딩 행렬을 계산 / 반환

    :param position: 임베딩 행렬의 총 열 개수
    :param d_model: 모델의 하이퍼파라미터
    :return: 포지셔널 인코딩 행렬
    """
    positional_encode_vector = np.zeros((position, d_model))

    for pos, row_vector in enumerate(positional_encode_vector):
        for i, element in enumerate(row_vector):
            row_vector[i] = _get_value(pos, i, d_model)

    return positional_encode_vector
