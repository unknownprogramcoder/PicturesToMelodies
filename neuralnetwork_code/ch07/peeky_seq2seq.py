# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from seq2seq import Seq2seq, Encoder

scale_range_size = 50
hidden_vector_size = 144

class PeekyDecoder:
    def __init__(self, scale_size=50, melvec_size=144, #embed affine가중치 공유하면 성능 좋다고 하네요
                 hidden_size=144, dropout_ratio=0.5):
        S, D, H = scale_size, melvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(S, D) / np.sqrt(S)).astype('f')
        lstm_Wx1 = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        lstm_Wx2 = (rn(H + H, 4 * H) / np.sqrt(H + H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')
        affine_W = (rn(H + H, S) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(S).astype('f')
        
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(affine_W, affine_b) 
        ]
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None

    def forward(self, xs, h, train_flg=True):
        N, T = xs.shape
        N, H = h.shape
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.lstm_layers:
            layer.set_state(h)
        
        hs = np.repeat(h, T, axis=0).reshape(N, T, H) #repeat노드 순전파

        out = self.layers[0].forward(xs)
        out = self.layers[1].forward(out)
        out = np.concatenate((hs, out), axis=2)
        out = self.layers[2].forward(out)
        out = self.layers[3].forward(out)
        out = np.concatenate((hs, out), axis=2)
        out = self.layers[4].forward(out)
        out = self.layers[5].forward(out)
        out = np.concatenate((hs, out), axis=2)
        score = self.layers[6].forward(out)
        self.cache = H
        return score

    def backward(self, dscore):
        H = self.cache

        dout = self.layers[6].backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.layers[5].backward(dout)
        dout = self.layers[4].backward(dout)
        dout, dhs1 = dout[:, :, H:], dout[:, :, :H]
        dout = self.layers[3].backward(dout)
        dout = self.layers[2].backward(dout)
        dout, dhs2 = dout[:, :, H:], dout[:, :, :H]
        dembed = self.layers[1].backward(dout)
        self.layers[0].backward(dembed)

        dhs = dhs0 + dhs1 + dhs2
        dh = self.lstm_layers[0].dh + self.lstm_layers[1].dh + np.sum(dhs, axis=1) #repeat 노드 역전파
        return dh 

    def generate(self, h, start_id, sample_size, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        sampled = []
        char_id = start_id
        for layer in self.lstm_layers:
            layer.set_state(h)
        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array(char_id).reshape((1, 1))
            hs = np.repeat(h, 1, axis=0).reshape(1, 1, H)
            out = self.layers[0].forward(x)
            out = self.layers[1].forward(out)
            out = np.concatenate((hs, out), axis=2)
            out = self.layers[2].forward(out)
            out = self.layers[3].forward(out)
            out = np.concatenate((hs, out), axis=2)
            out = self.layers[4].forward(out)
            out = self.layers[5].forward(out)
            out = np.concatenate((hs, out), axis=2)
            score = self.layers[6].forward(out)
            char_id = np.argmax(score.flatten())
            sampled.append(char_id)
        return sampled


class PeekySeq2seq(Seq2seq):
    def __init__(self, scale_size=scale_range_size, melvec_size=hidden_vector_size, hidden_size=hidden_vector_size):
        S, D, H = scale_size, melvec_size, hidden_size
        self.encoder = Encoder()
        self.decoder = PeekyDecoder(S, D, H)
        self.softmax = TimeSoftmaxWithLoss()
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
