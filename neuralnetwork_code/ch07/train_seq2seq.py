# coding: utf-8
import sys
sys.path.append('..')
from common.np import *
import numpy as rnp #real num py. you know what i'm saying?
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq
import pickle

# 데이터셋 읽기
with open('../pictures_for_encoder_input_train', 'rb') as f:
    x_train = pickle.load(f)
with open('../melodies_for_decoder_input_train', 'rb') as f:
    t_train = pickle.load(f)
t_train = t_train.astype(int)
# 하이퍼파라미터 설정
batch_size = 1 #gpu최대 처리 용량 한계로 batch_size는 4가 최대. ㅠㅠ
max_epoch = 500
max_grad = 5.0

# 일반 혹은 엿보기(Peeky) 설정 =====================================
#model = Seq2seq()
model = PeekySeq2seq()
# ================================================================
optimizer = Adam()
trainer = Trainer(model, optimizer)

for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)
model.save_params(file_name='seq2seq_parameters')
# 그래프 그리기

x = rnp.arange(len(trainer.loss_list))
plt.plot(x, trainer.loss_list, marker='o')
plt.xlabel('count')
plt.ylabel('average_loss')
plt.show()