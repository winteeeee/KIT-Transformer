import tensorflow as tf


def position_wise_ffnn(d_ff_dense, d_model_dense, attention):
    """
    행렬에 비선형성 추가, 다양한 특징 학습, 모델 용량 증가 등 다양한 역할 수행

    두 개의 Dense층으로 구성되어 있으며 첫 Dense는 d_ff개의 은닉층을 지니도록 하고
    이후 벡터가 또 다른 인코더의 입력으로 들어가야 하므로 두 번째 Dense는 d_model개의 은닉층을 지니도록 함

    :param d_ff_dense: Dense(d_ff)
    :param d_model_dense: Dense(d_model)
    :param attention: 어텐션 행렬
    """

    ffnn_vector = d_ff_dense(attention)
    ffnn_vector = d_model_dense(ffnn_vector)
    return ffnn_vector
