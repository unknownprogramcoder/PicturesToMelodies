# coding: utf-8
import sys
sys.path.append('..')
import pickle
from common.np import *
from collections import OrderedDict
from common.layers import *
from common.time_layers import *
from common.base_model import BaseModel

# 전국과학전람회 연구에 사용될 핵심 인공신경망.
image_channels = 1
image_size_height = 256
image_size_width = 256
scale_range_size = 50
hidden_vector_size = 144
class Encoder: #cnn신경망, deep_convnet들고옴.
    def __init__(self, input_dim=(image_channels, image_size_height, image_size_width),
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=256, output_size=hidden_vector_size):
        # 가중치 초기화===========
        # 각 층의 뉴런 하나당 앞 층의 몇 개 뉴런과 연결되는가（TODO: 자동 계산되게 바꿀 것）
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값
        self.params = {}
        pre_channel_num = input_dim[0]
        
        W1 = wight_init_scales[0] * np.random.randn(conv_param_1['filter_num'], pre_channel_num, conv_param_1['filter_size'], conv_param_1['filter_size'])
        b1 = np.zeros(conv_param_1['filter_num'])
        pre_channel_num = conv_param_1['filter_num']
        
        W2 = wight_init_scales[1] * np.random.randn(conv_param_2['filter_num'], pre_channel_num, conv_param_2['filter_size'], conv_param_2['filter_size'])
        b2 = np.zeros(conv_param_2['filter_num'])
        pre_channel_num = conv_param_2['filter_num']

        W3 = wight_init_scales[2] * np.random.randn(conv_param_3['filter_num'], pre_channel_num, conv_param_3['filter_size'], conv_param_3['filter_size'])
        b3 = np.zeros(conv_param_3['filter_num'])
        pre_channel_num = conv_param_3['filter_num']

        W4 = wight_init_scales[3] * np.random.randn(conv_param_4['filter_num'], pre_channel_num, conv_param_4['filter_size'], conv_param_4['filter_size'])
        b4 = np.zeros(conv_param_4['filter_num'])
        pre_channel_num = conv_param_4['filter_num']

        W5 = wight_init_scales[4] * np.random.randn(conv_param_5['filter_num'], pre_channel_num, conv_param_5['filter_size'], conv_param_5['filter_size'])
        b5 = np.zeros(conv_param_5['filter_num'])
        pre_channel_num = conv_param_5['filter_num']

        W6 = wight_init_scales[5] * np.random.randn(conv_param_6['filter_num'], pre_channel_num, conv_param_6['filter_size'], conv_param_6['filter_size'])
        b6 = np.zeros(conv_param_6['filter_num'])
        pre_channel_num = conv_param_6['filter_num']

        W7 = wight_init_scales[6] * np.random.randn(64*32*32, hidden_size)
        b7 = np.zeros(hidden_size)
        W8 = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        b8 = np.zeros(output_size)

        # 계층 생성===========
        self.layers = []
        self.layers.append(Convolution(W1, b1, conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(W2, b2, conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(W3, b3, conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(W4, b4, conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(W5, b5, conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(W6, b6, conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(W7, b7))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(W8, b8))
        self.layers.append(Dropout(0.5))

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def forward(self, x, train_flg=True):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def backward(self, dh):
        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dh = layer.backward(dh)
        dout = dh
        return dout

class Decoder: #rnn신경망
    def __init__(self, scale_size=scale_range_size, melvec_size=hidden_vector_size, #embed affine가중치 공유하면 성능 좋다고 하네요
                 hidden_size=hidden_vector_size, dropout_ratio=0.5):
        S, D, H = scale_size, melvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(S, D) / np.sqrt(S)).astype('f')
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        #lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        #lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        #lstm_b2 = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, S) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(S).astype('f')
        
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            #TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            #TimeDropout(dropout_ratio),
            TimeAffine(affine_W, affine_b) 
        ]
        self.lstm_layers = [self.layers[2]]#, self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3]]#, self.layers[5]]
        
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h, train_flg=True):
        print("sdlfaj", xs.shape)
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.lstm_layers:
            layer.set_state(h)
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def backward(self, dscore):
        for layer in reversed(self.layers):
            dscore = layer.backward(dscore)
        dh = 0
        for layer in self.lstm_layers:
            dh += layer.dh   #repeat 노드의 역전파: 기울기를 다 더함.
        return dh

    def generate(self, h, start_id=32, sample_size=256, train_flg=False): #start_id 예시 32:c 34:d .....
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        sampled = []
        sample_id = start_id
        for layer in self.lstm_layers:
            layer.set_state(h)
        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            for layer in self.layers:
                x = layer.forward(x)
            score = x
            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))
        return sampled

class Seq2seq(BaseModel): #BaseModel이 상위 클래스, 여기에서 매개변수 저장 및 불러오기 기능 받아옴.
    def __init__(self, scale_size=scale_range_size, melvec_size=hidden_vector_size, hidden_size=hidden_vector_size):
        S, D, H = scale_size, melvec_size, hidden_size
        self.encoder = Encoder() 
        self.decoder = Decoder(S, D, H)
        self.softmax = TimeSoftmaxWithLoss() #손실 계산
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs_enc, xs_dec):
        decoder_xs, decoder_ts = xs_dec[:, :-1], xs_dec[:, 1:] #어디서 받아오는지 모르겠다
        h = self.encoder.forward(xs_enc)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size=256):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled

