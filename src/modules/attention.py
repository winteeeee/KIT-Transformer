import tensorflow as tf


def _scaled_dot_product_attention(query, key, value, mask):
    """
    스케일링 기능을 추가한 닷 프로덕트 어텐션 함수

    :param query: 쿼리 행렬
    :param key: 키 행렬
    :param value: 값 행렬
    :param mask: 마스킹 행렬
    :return: softmax(QK^T / √d_k)V
    """

    d_k = tf.cast(key.shape[-1], tf.float32)
    attention_score_mat = tf.matmul(query, key, transpose_b=True) / tf.sqrt(d_k)
    if mask is not None:
        attention_score_mat += mask * -1e9

    return tf.matmul(tf.nn.softmax(attention_score_mat), value)


def create_pad_mask(inputs):
    """
    패딩 마스크 생성.
    입력 벡터에 패딩 토큰이 있을 경우(값이 0) 해당 위치를 1로 표기
    :param inputs: 입력 벡터
    :return: 패딩 마스크
    """
    pad_mask = tf.cast(tf.where(tf.equal(inputs, 0), 1, 0), tf.float32)
    return pad_mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(inputs):
    """
    룩 어헤드 마스크 생성.
    디코더의 첫 번째 서브층에서 미래 단어를 참조하지 못하도록 해당 위치를 1로 표기
    :param inputs:
    :return:
    """
    inputs_len = inputs.shape[1]
    # tensorflow의 삼각행렬 생성 함수
    look_ahead_mask = tf.linalg.band_part(tf.ones((inputs_len, inputs_len)), 0, -1)
    pad_mask = create_pad_mask(inputs)
    return tf.maximum(look_ahead_mask, pad_mask)


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


def multi_head_attention(dense_layer, query, key, value, d_model, num_heads, mask=None):
    """
    멀티-헤드 어텐션 수행

    :param dense_layer: Dense(d_model)
    :param query: 쿼리 행렬
    :param key: 키 행렬
    :param value: 값 행렬
    :param d_model: 모델의 하이퍼파라미터
    :param num_heads: 병렬 연산 수
    :param mask: 스케일드 닷-프로덕트 어텐션 수행 시 마스크 적용 여부 결정(기본값 None)
    :return: 멀티-헤드 어텐션 적용 행렬
    """
    size_per_head = d_model // num_heads
    batch_size = tf.shape(query)[0]

    w_q = dense_layer
    w_k = dense_layer
    w_v = dense_layer
    w_o = dense_layer

    query = _split_matrix(matrix=w_q(query), num_heads=num_heads, size_per_head=size_per_head, batch_size=batch_size)
    key = _split_matrix(matrix=w_k(key), num_heads=num_heads, size_per_head=size_per_head, batch_size=batch_size)
    value = _split_matrix(matrix=w_v(value), num_heads=num_heads, size_per_head=size_per_head, batch_size=batch_size)

    attention = _scaled_dot_product_attention(query=query, key=key, value=value, mask=mask)
    # (batch_size, num_heads, seq_len, size_per_head) -> (batch_size, seq_len, num_head, size_per_head)
    # 어텐션 때문에 바꿔두었던 차원 정상화
    attention = tf.transpose(attention, perm=[0, 2, 1, 3])

    # split 해두었던 행렬 합치기
    concat = tf.reshape(attention, (batch_size, -1, d_model))
    return w_o(concat)
