# [KIT-Transformer]
## 1. 개요
KIT-Transformer는 금오공과대학교 지식언어공학 연구실의 2023년도 동계 SIG 활동으로 진행한 개인 프로젝트이다.

## 2. 목적
지금까지 블랙박스처럼 사용해왔던 트랜스포머 모델의 원리, 구조 등을 이해하기 위해 논문 및 기타 레퍼런스를 참고하여 트랜스포머를 직접 구현해본다.

또한 이를 바탕으로 추후 후배들도 보고 참고할 수 있을만한 기록을 남기는 것을 목적으로 한다. 이를 위해서 텐서플로우의 Subclassing API를 이용하고 논문에서 제시된 구조를 코드 상으로 확실히 알아볼 수 있게 높은 가독성을 지닌 코드 작성을 지향한다.

## 3. 사용 라이브러리
Tensorflow 2.15.0

## 4. 프로젝트 구조
```
* KIT-Transformer: 프로젝트 루트
┣ * src: 소스 루트
┃ ┣ * modules: 트랜스포머 내부 모듈 폴더
┃ ┃ ┣ * add_and_norm: Add & Norm 모듈
┃ ┃ ┃  
┃ ┃ ┣ * attention: Multi-Head Attention 모듈
┃ ┃ ┃  
┃ ┃ ┣ * decoder: 트랜스포머 디코더
┃ ┃ ┃  
┃ ┃ ┣ * embedding: Embedding 모듈
┃ ┃ ┃  
┃ ┃ ┣ * encoder: 트랜스포머 인코더
┃ ┃ ┃  
┃ ┃ ┣ * position_wise_FFNN: Feed Forward 모듈
┃ ┃ ┃  
┃ ┃ ┗ * positional_encoder: Positinal Encoding 모듈
┃ ┃
┃ ┣ * chatbot_example: 챗봇 예제
┃ ┃
┃ ┣ * classification_example: 텍스트 분류 예제
┃ ┃
┃ ┣ * scheduler: 트랜스포머 학습 스케쥴러
┃ ┃
┃ ┗ * transformer: 트랜스포머 모델
┃
┣ * .gitignore: 프로젝트 gitignore 파일
┃ 
┣ * paper_kr: Attention Is All You Need 한국어 번역본
┃ 
┣ * README: 프로젝트 README 파일
┃ 
┗ * requirements: 프로젝트 requirements 파일
```

## 5. 사용 방법
### 5-1. 트랜스포머
transformer.py의 Transformer를 import하고 하이퍼파라미터를 지정하여 인스턴스화 시킨 다음 fit 한다.

입력 모양은 (2, batch_size, seq_len) 꼴이 되어야 하고 첫 번째 차원의 0번째 원소는 인코더의 입력, 1번째 원소는 디코더의 입력을 넣어준다.

call 함수 내부에서 자동으로 임베딩과 포지셔널 인코딩을 수행하기 때문에 데이터만 넣어주고 fit을 호출하면 모든 과정이 자동으로 수행된다.

마지막 레이어에 Softmax 함수는 적용되어 있지 않고 완전연결층만 존재하기 때문에 필요 시 함수의 출력 이후 Softmax층을 추가할 수 있다. 

구체적인 사용 예시는 chatbot_example.ipynb 참고.

### 5-2. 트랜스포머 인코더
시퀀스의 생성이 아닌 텍스트 분류와 같은 태스크의 경우 인코더-디코더 구조가 아닌 트랜스포머의 인코더만을 가져와서 순환 신경망 대용으로 사용할 수 있다.

이 경우 embedding과 positional encoding 작업을 수동으로 하여야 한다(위 모듈을 encoder 내부로 옮기면 자동으로 수행하도록 할 수도 있지만 코드를 보고 논문 내부 구조를 직관적으로 떠올릴 수 있도록 의도적으로 외부로 분리하였음)

encoder.py의 Encoder, embedding.py의 embedding, positional_encoder.py의 positional_encoding을 import하고 입력 데이터가 들어오면 embedding -> positional_encoding -> encode의 순서대로 순방향 pass가 진행되도록 설계하면 되며 태스크에 따라 추가적인 층을 추가할 수 있다.

구체적인 사용 예시는 classification_example.ipynb 참고.

### 6. 모듈 정보
#### Embedding
```python
def embedding(embedding_layer, inputs, d_model):
    """
    keras의 Embedding층을 이용해 입력을 임베딩하고 가중치를 √d_model와 곱한 임베딩 행렬을 반환

    :param embedding_layer: 임베딩에 사용할 임베딩 레이어
    :param inputs: 모델의 입력
    :param d_model: 모델의 하이퍼파라미터
    :return: 입력을 임베딩한 행렬
    """
    embedding_vector = embedding_layer(inputs)
    embedding_vector *= tf.math.sqrt(tf.cast(d_model, tf.float32))

    return embedding_vector
```
임베딩은 특출난 것 없이 keras의 Embedding 레이어를 사용합니다.

#### Positinal Encoding
```python
def positional_encoding(dropout_layer, inputs, position, d_model):
    """
    임베딩 행렬을 입력받아 포지셔널 인코딩 행렬을 더하여 인코더, 디코더의 입력으로 사용할 수 있는 임베딩 행렬 반환
    기본적으로 드롭아웃은 서브층 이후에 적용되지만 예외적으로 임베딩 벡터 + 포지셔널 인코딩 작업 이후에도 적용

    :param dropout_layer: Dropout에 사용할 Dropout 레이어
    :param inputs: 임베딩 행렬
    :param position: 입력의 크기
    :param d_model: 모델의 하이퍼파라미터
    :return: 포지셔널 인코딩 행렬이 더해진 임베딩 행렬
    """
    positional_encode_mat = _get_positional_encode_matrix(position, d_model)
    if positional_encode_mat is not None:
        inputs += positional_encode_mat

    return dropout_layer(inputs)
```
모델에게 토큰의 위치 정보를 강제로 삽입하기 위한 인코딩 기법입니다. 임베딩 행렬과 차원이 동일한 행렬이 되도록 하여 임베딩 행렬과 더하기 연산을 통해 위치 정보를 주입합니다.

포지셔널 행렬의 값은 삼각함수의 값으로 계산합니다. 삼각 함수는 주기성과 연속성을 만족하고 있기 때문에 위치 정보를 의미하기에 적합합니다.

#### Position-wise Feed Forward Neuron Network
```python
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
```
Feed Forward Neuron Network는 회귀하지 않는 일반 완전연결층을 의미합니다. 행렬에 비선형성추가, 다양한 특징 학습, 모델 용량 증가 등 다양한 이유에서 사용됩니다.

#### Add & Norm
```python
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
```
논문에서는 서브층의 출력 뒤에 잔차 연결 + 정규화 층을 붙입니다. 이는 수식으로 LayerNorm(x + Sublayer(x))로 표현됩니다. 

x + Sublayer(x)의 의미는 입력값과 서브층의 출력 값을 더하라는 의미로, 이를 잔차 연결이라고 합니다. 잔차 연결은 본래 Deep CNN 구조에서 처음 제시된 개념으로 깊이가 깊어질 수록 feature가 너무 많아져 도리어 shallow한 모델보다 성능이 하락하는 현상을 막고자 고안되었습니다. 잔차 연결을 통해 전체 값을 학습하는 것이 아닌, 입력과 출력의 차이만을 학습합니다.

층 정규화는 Scale의 영향을 최소화하여 안정적인 학습을 위해 사용합니다.

#### Attention
```python
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
```

순환 신경망의 단점은 정보 손실과 기울기 소실을 개선하기 위해 처음 등장한 개념입니다. 본래 순환 신경망의 성능을 높이기 위해 고안되었지만 트랜스포머에서는 오직 어텐션만으로만 구성됩니다.

어텐션에는 Query, Key, Value의 값이 존재하며, Query와 Key 사이의 유사도를 구한 뒤 Softmax 함수를 적용시켜 Attetion Distribution을 얻습니다.

Attention Distribution의 값과 value 값을 곱해 어떤 value에 핵심적으로 더 **집중** 해야 하는지 구할 수 있습니다.

트랜스포머에서는 이를 Scaled Dot-Product Attetion으로 구현하고 있습니다.

#### Multi-Head Attention
```python
def multi_head_attention(w_q, w_k, w_v, w_o, query, key, value, d_model, num_heads, mask=None):
    """
    멀티-헤드 어텐션 수행

    :param w_q: Dense(d_model)
    :param w_k: Dense(d_model)
    :param w_v: Dense(d_model)
    :param w_o: Dense(d_model)
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
```
입력 행렬을 쪼갠 후 어텐션을 병렬적으로 수행하는 것을 의미합니다. 각각의 헤드마다 다른 시점에서 폭넓은 정보를 수집할 수 있게 됩니다.

쪼개진 행렬은 추후 concatenate하여 본래의 행렬로 복원합니다.

## 7. 참고 문헌
1. 유원준, 안상준, 「딥 러닝을 이용한 자연어 처리 입문」
2. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, 「Attention Is All You Need」
