from modules.encoder import Encoder
from decoder import Decoder


class Transformer:
    def __init__(self, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1):
        """
        :param d_model: 인코더와 디코더의 입출력 크기 및 임베딩 차원을 설정. 기본값 512
        :param num_layers: 인코더와 디코더의 층 수. 기본값 6
        :param num_heads: 병렬 어텐션 개수. 기본값 8
        :param dropout: Dropout층의 Rate
        :param d_ff: 내부 피드 포워드 신경망의 은닉층 크기. 기본값 2048
        """
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.encoder = Encoder(d_model=d_model, num_heads=num_heads, dropout=dropout, d_ff=d_ff)
        self.decoder = Decoder()
