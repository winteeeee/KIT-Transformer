import tensorflow as tf


def embedding(embedding_layer, inputs, d_model):
    """
    keras의 Embedding층을 이용해 입력을 임베딩하고 가중치를 √d_model와 곱한 임베딩 행렬을 반환

    :param embedding_layer: 임베딩에 사용할 임베딩 레이어
    :param inputs: 모델의 입력
    :param d_model: 모델의 하이퍼파라미터
    :return: 입력을 임베딩한 행렬
    """
    embedding_vector = embedding_layer(inputs)
    embedding_vector *= tf.math.sqrt(tf.cast(d_model, tf.float32))

    return embedding_vector
