from tensorflow.keras.layers import Dense


def position_wise_FFNN(attention, d_model, d_ff):
    """
    행렬에 비선형성 추가, 다양한 특징 학습, 모델 용량 증가 등 다양한 역할 수행

    두 개의 Dense층으로 구성되어 있으며 첫 Dense는 d_ff개의 은닉층을 지니도록 하고
    이후 벡터가 또 다른 인코더의 입력으로 들어가야 하므로 두 번째 Dense는 d_model개의 은닉층을 지니도록 함
    :param attention: 어텐션 행렬
    :param d_model: 입력 차원
    :param d_ff: 피드 포워드 신경망 은닉 차원
    """

    FFNN_vector = Dense(d_ff, activation='relu')(attention)
    FFNN_vector = Dense(d_model)(FFNN_vector)
    return FFNN_vector