from attention import multi_head_attention, create_pad_mask
from add_and_norm import add_and_norm
from position_wise_FFNN import position_wise_FFNN


class Encoder:
    def __init__(self, d_model, num_heads, dropout, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_ff = d_ff

    def encode(self, inputs):
        """
        인코딩 수행

        :param inputs: 임베딩 + 포지셔널 인코딩 시퀀스
        :return: 인코더층을 지난 시퀀스
        """
        attention = multi_head_attention(query=inputs, key=inputs, value=inputs,
                                         d_model=self.d_model,
                                         num_heads=self.num_heads,
                                         mask=create_pad_mask(inputs))
        attention = add_and_norm(inputs=inputs, outputs=attention, dropout=self.dropout)
        ffnn = position_wise_FFNN(attention=attention, d_model=self.d_model, d_ff=self.d_ff)
        ffnn = add_and_norm(inputs=attention, outputs=ffnn, dropout=self.dropout)

        return ffnn
