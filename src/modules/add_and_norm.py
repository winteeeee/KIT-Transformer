def _residual_connect(inputs, outputs):
    """
    서브층의 입력과 출력을 더하여 x + Sublayer(x)를 구현
    차원은 d_model로 동일하므로 덧셈 가능

    :param inputs: 서브층의 입력
    :param outputs: 서브층의 출력
    :return: x + Sublayer(x)
    """
    return inputs + outputs


def _layer_normalize(normalization_layer, layer):
    """
    텐서의 마지막 차원에 대해 평균과 분산을 구하여 층을 정규화
    keras의 LayerNormalization() 사용

    :param normalization_layer: LayerNormalization 레이어
    :param layer: 정규화할 층
    :return: 정규화된 층
    """
    return normalization_layer(layer)


def add_and_norm(dropout_layer, normalization_layer, inputs, outputs):
    """
    트랜스포머의 Add & Norm층 구현

    :param dropout_layer: Dropout 레이어
    :param normalization_layer: LayerNormalization 레이어
    :param inputs: 하위층 입력
    :param outputs: 하위층 출력
    :return: Add & Norm 적용 결과
    """
    outputs = dropout_layer(outputs)
    return _layer_normalize(normalization_layer, _residual_connect(inputs, outputs))
