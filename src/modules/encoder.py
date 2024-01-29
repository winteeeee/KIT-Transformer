from modules.add_and_norm import add_and_norm
from modules.position_wise_FFNN import position_wise_ffnn
from modules.attention import multi_head_attention

from tensorflow.keras.layers import Dense, Dropout, LayerNormalization


class Encoder:
    def __init__(self, d_model, num_heads, dropout, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_ff = d_ff

        self.w_q = Dense(d_model)
        self.w_k = Dense(d_model)
        self.w_v = Dense(d_model)
        self.w_o = Dense(d_model)
        self.d_ff_dense = Dense(d_ff)
        self.d_model_dense = Dense(d_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)


    def encode(self, inputs, pad_mask=None):
        """
        인코딩 수행

        :param inputs: 임베딩 + 포지셔널 인코딩 시퀀스
        :param pad_mask: Multi-Head Attention에서 사용될 패딩 mask 행렬
        :return: 인코더층을 지난 시퀀스
        """
        sublayer1 = multi_head_attention(w_q=self.w_q, w_k=self.w_k, w_v=self.w_v, w_o=self.w_o,
                                         query=inputs, key=inputs, value=inputs,
                                         d_model=self.d_model,
                                         num_heads=self.num_heads,
                                         mask=pad_mask)
        sublayer1 = add_and_norm(dropout_layer=self.dropout1,
                                 normalization_layer=self.norm1,
                                 inputs=inputs, outputs=sublayer1)

        sublayer2 = position_wise_ffnn(d_ff_dense=self.d_ff_dense, d_model_dense=self.d_model_dense,
                                       attention=sublayer1)
        sublayer2 = add_and_norm(dropout_layer=self.dropout2,
                                 normalization_layer=self.norm2,
                                 inputs=sublayer1, outputs=sublayer2)

        return sublayer2
