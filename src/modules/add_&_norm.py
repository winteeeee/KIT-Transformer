from tensorflow.keras.layers import LayerNormalization


def _residual_connect(inputs, outputs):
    """
    서브층의 입력과 출력을 더하여 x + Sublayer(x)를 구현
    차원은 d_model로 동일하므로 덧셈 가능

    :param inputs: 서브층의 입력
    :param outputs: 서브층의 출력
    :return: x + Sublayer(x)
    """
    return inputs + outputs


def _layer_normalize(layer, epsilon=1e-6):
    """
    텐서의 마지막 차원에 대해 평균과 분산을 구하여 층을 정규화
    keras의 LayerNormalization() 사용

    :param layer: 정규화할 층
    :param epsilon: LayerNormliazation의 하이퍼파라미터. 분모가 0이 되지 않도록 함
    :return: 정규화된 층
    """
    return LayerNormalization(epsilon=epsilon)(layer)


def add_and_norm(inputs, outputs):
    """
    트랜스포머의 Add & Norm층 구현

    :param inputs: 하위층 입력
    :param outputs: 하위층 출력
    :return: Add & Norm 적용 결과
    """
    return _layer_normalize(_residual_connect(inputs, outputs))
