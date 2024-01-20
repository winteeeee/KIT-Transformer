import tensorflow as tf
from tensorflow.keras.layers import Embedding


def embedding(inputs, vocab_size, d_model):
    """
    keras의 Embedding층을 이용해 입력을 임베딩하고 가중치를 √d_model와 곱한 임베딩 행렬을 반환

    :param inputs: 모델의 입력
    :param vocab_size: 입력에서의 총 vocab 크기
    :param d_model: 모델의 하이퍼파라미터
    :return: 입력을 임베딩한 행렬
    """
    embedding_vector = Embedding(vocab_size, d_model)(inputs)
    embedding_vector *= tf.math.sqrt(tf.cast(d_model, tf.float32))

    return embedding_vector
