import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


def _scaled_dot_product_attention(query, key, value):
    """
    스케일링 기능을 추가한 닷 프로덕트 어텐션 함수

    :param query: 쿼리 행렬
    :param key: 키 행렬
    :param value: 값 행렬
    :return: softmax(QK^T / √d_k)V
    """
    key_t = np.transpose(key)
    d_k = key_t.shape[0]
    return np.matmul(tf.nn.softmax((np.matmul(query, key_t)) / np.sqrt(d_k)), value)


def _split_matrix(matrix, num_heads, size_per_head, batch_size):
    """
    행렬을 병렬 연산 수 만큼 스플릿
    d_model을 num_heads와 size_per_head로 나누고 차원을 추가
    이때 스케일드 닷 프로덕트 어텐션의 벡터는 seq_len, d_k(size_per_head)가 입력으로 들어와야 하므로
    transpose를 통해 차원 조정
    (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, size_per_head)

    :param matrix: 스플릿 대상 행렬
    :param num_heads: 병렬 연산 수
    :param size_per_head: 병렬 연산에 맞춰 나눠지는 행렬 크기
    :param batch_size: 모델 학습 배치 크기
    :return: 헤드 별로 나눠진 행렬
    """
    matrix = tf.reshape(matrix, shape=(batch_size, -1, num_heads, size_per_head))
    return tf.transpose(matrix, perm=[0, 2, 1, 3])


def multi_head_attention(query, key, value, d_model, num_heads):
    """
    멀티-헤드 어텐션 수행

    :param query: 쿼리 행렬
    :param key: 키 행렬
    :param value: 값 행렬
    :param d_model: 모델의 하이퍼파라미터
    :param num_heads: 병렬 연산 수
    :return: 멀티-헤드 어텐션 적용 행렬
    """
    size_per_head = d_model // num_heads
    batch_size = query.shape[0]

    w_q = Dense(d_model)
    w_k = Dense(d_model)
    w_v = Dense(d_model)
    w_o = Dense(d_model)

    query = _split_matrix(matrix=w_q(query), num_heads=num_heads, size_per_head=size_per_head, batch_size=batch_size)
    key = _split_matrix(matrix=w_k(key), num_heads=num_heads, size_per_head=size_per_head, batch_size=batch_size)
    value = _split_matrix(matrix=w_v(value), num_heads=num_heads, size_per_head=size_per_head, batch_size=batch_size)

    attention = _scaled_dot_product_attention(query=query, key=key, value=value)
    # (batch_size, num_heads, seq_len, size_per_head) -> (batch_size, seq_len, num_head, size_per_head)
    # 어텐션 때문에 바꿔두었던 차원 정상화
    attention = tf.transpose(attention, perm=[0, 2, 1, 3])

    # split 해두었던 행렬 합치기
    concat = tf.reshape(attention, (batch_size, -1, d_model))
    return w_o(concat)
