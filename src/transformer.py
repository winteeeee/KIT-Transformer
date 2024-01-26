from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.embedding import embedding
from modules.positional_encoder import positional_encoding
from modules.attention import create_pad_mask, create_look_ahead_mask

import tensorflow as tf
from tensorflow.keras.layers import Dense


class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1):
        """
        :param vocab_size: 학습 데이터의 단어 크기
        :param d_model: 인코더와 디코더의 입출력 크기 및 임베딩 차원을 설정. 기본값 512
        :param num_layers: 인코더와 디코더의 층 수. 기본값 6
        :param num_heads: 병렬 어텐션 개수. 기본값 8
        :param dropout: Dropout층의 Rate
        :param d_ff: 내부 피드 포워드 신경망의 은닉층 크기. 기본값 2048
        """
        super(Transformer, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.encoder = Encoder(d_model=d_model, num_heads=num_heads, dropout=dropout, d_ff=d_ff)
        self.decoder = Decoder(d_model=d_model, num_heads=num_heads, dropout=dropout, d_ff=d_ff)

    def call(self, inputs):
        encoder_inputs = inputs[0]
        decoder_inputs = inputs[1]
        encoder_mask = create_pad_mask(encoder_inputs)
        decoder_pad_mask = create_pad_mask(decoder_inputs)
        decoder_look_ahead_mask = create_look_ahead_mask(decoder_inputs)

        encoder_inputs = embedding(inputs=encoder_inputs, vocab_size=self.vocab_size, d_model=self.d_model)
        encoder_inputs = positional_encoding(inputs=encoder_inputs,
                                             vocab_size=self.vocab_size,
                                             d_model=self.d_model,
                                             dropout=self.dropout)

        encoder_outputs = self.encoder.encode(inputs=encoder_inputs, pad_mask=encoder_mask)
        for i in range(self.num_layers - 1):
            encoder_outputs = self.encoder.encode(inputs=encoder_outputs)

        decoder_inputs = embedding(inputs=decoder_inputs, vocab_size=self.vocab_size, d_model=self.d_model)
        decoder_inputs = positional_encoding(inputs=decoder_inputs,
                                             vocab_size=self.vocab_size,
                                             d_model=self.d_model,
                                             dropout=self.dropout)

        decoder_outputs = self.decoder.decode(inputs=decoder_inputs, encoder_outputs=encoder_outputs,
                                              pad_mask=decoder_pad_mask, look_ahead_mask=decoder_look_ahead_mask)
        for i in range(self.num_layers - 1):
            decoder_outputs = self.decoder.decode(inputs=decoder_outputs, encoder_outputs=encoder_outputs)

        outputs = Dense(units=self.vocab_size)(decoder_outputs)
        return tf.nn.softmax(outputs)
