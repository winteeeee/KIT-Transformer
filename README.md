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

## 6. 참고 문헌
1. 유원준, 안상준, 「딥 러닝을 이용한 자연어 처리 입문」
2. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, 「Attention Is All You Need」