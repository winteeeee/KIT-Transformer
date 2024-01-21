from add_and_norm import add_and_norm
from position_wise_FFNN import position_wise_FFNN
from attention import multi_head_attention, create_look_ahead_mask, create_pad_mask


class Decoder:
    def __init__(self, d_model, num_heads, dropout, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_ff = d_ff

    def decode(self, inputs, encoder_outputs):
        """
        디코딩 수행

        :param inputs: 임베딩 + 포지셔널 인코딩 시퀀스
        :param encoder_outputs: 인코더층을 지난 시퀀스
        :return: 디코딩된 시퀀스
        """
        sublayer1 = multi_head_attention(query=inputs, key=inputs, value=inputs,
                                         d_model=self.d_model,
                                         num_heads=self.num_heads,
                                         mask=create_look_ahead_mask(inputs))
        sublayer1 = add_and_norm(inputs=inputs, outputs=sublayer1, dropout=self.dropout)

        sublayer2 = multi_head_attention(query=sublayer1, key=encoder_outputs, value=encoder_outputs,
                                         d_model=self.d_model,
                                         num_heads=self.num_heads,
                                         mask=create_pad_mask(inputs))
        sublayer2 = add_and_norm(inputs=sublayer1, outputs=sublayer2, dropout=self.dropout)

        sublayer3 = position_wise_FFNN(attention=sublayer2, d_model=self.d_model, d_ff=self.d_ff)
        sublayer3 = add_and_norm(inputs=sublayer2, outputs=sublayer3, dropout=self.dropout)

        return sublayer3
