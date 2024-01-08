import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout
from positional_encoder import positional_encode


def embedding(inputs, vocab_size, d_model, dropout):
    embedding_vector = Embedding(vocab_size, d_model)(inputs)
    embedding_vector *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embedding_vector += positional_encode(vocab_size, d_model)
    # 기본적으로 드롭아웃은 서브층 이후에 적용되지만 예외적으로 임베딩 벡터 + 포지셔널 인코딩 작업 이후에도 적용
    return Dropout(rate=dropout)(embedding_vector)