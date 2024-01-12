import numpy as np
import tensorflow as tf


def scaled_dot_product_attention(query, key, value):
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
