# coding: utf-8
import sys
sys.path.append('..')
from common.np import *
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
with open('../pictures_for_encoder_input_test', 'rb') as f:
    x_test = pickle.load(f)
with open('../melodies_for_decoder_input_test', 'rb') as f:
    t_test = pickle.load(f) 
    
t_train = t_train.astype(int)
t_test = t_test.astype(int)
# 하이퍼파라미터 설정
batch_size = 2 #>??
max_epoch1 = 1
max_grad = 5.0

# 일반 혹은 엿보기(Peeky) 설정 =====================================
#model = Seq2seq()
model = PeekySeq2seq()
# ================================================================
optimizer = Adam()
trainer = Trainer(model, optimizer)

loss_list = []
for epoch in range(max_epoch1):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)
    
    correct_num = 0
    total_loss = 0
    loss_count = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        loss = model.forward(question, correct)
        total_loss += loss
        loss_count += 1
    avg_loss = total_loss / loss_count
    print('시험 평균 오차', avg_loss)
    loss_list.append(avg_loss)

model.save_params(file_name='seq2seq_parameters')
# 그래프 그리기
x = np.arange(len(loss_list))
plt.plot(x, loss_list, marker='o')
plt.xlabel('에폭')
plt.ylabel('퍼플렉시티')
plt.ylim(0, 1.0)
plt.show()