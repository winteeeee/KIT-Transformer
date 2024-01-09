import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout
from positional_encoder import positional_encode


def embedding(inputs, vocab_size, d_model, dropout):
    """
    keras의 Embedding층을 이용해 입력을 임베딩하고 가중치를 √d_model와 곱한 후 포지셔널 인코딩 행렬과 더한, 최종적으로 트랜스포머가 사용할 임베딩 행렬을 반환

    :param inputs: 모델의 입력
    :param vocab_size: 입력에서의 총 vocab 크기
    :param d_model: 모델의 하이퍼파라미터
    :param dropout: 모델의 하이퍼파라미터
    :return: 입력을 임베딩한 행렬
    """
    embedding_vector = Embedding(vocab_size, d_model)(inputs)
    embedding_vector *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embedding_vector += positional_encode(vocab_size, d_model)
    # 기본적으로 드롭아웃은 서브층 이후에 적용되지만 예외적으로 임베딩 벡터 + 포지셔널 인코딩 작업 이후에도 적용
    return Dropout(rate=dropout)(embedding_vector)
