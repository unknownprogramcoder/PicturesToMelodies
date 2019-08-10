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
from matplotlib import pyplot as plt

# 데이터셋 읽기
'''
with open('../pictures_for_encoder_input_train', 'rb') as f:
    x_train = pickle.load(f)
with open('../pictures_for_encoder_input_test', 'rb') as f:
    x_test = pickle.load(f)
'''
with open('../pictures_for_encoder_input_new', 'rb') as f:
    x_new = pickle.load(f)
'''
t_train = t_train.astype(int)
t_test = t_test.astype(int)
'''
x_new = x_new.astype(int)
# 일반 혹은 엿보기(Peeky) 설정 =====================================
#model = Seq2seq()
model = PeekySeq2seq()
model.load_params(file_name='seq2seq_parameters')
# ================================================================

sample_size=256
scale_range_size = 50
start_id = 31
 #도.
for i in range(len(x_new)):
    x = x_new[i]
    x = np.expand_dims(x, axis=0)
    c = model.generate(x, start_id, sample_size=sample_size)
    a = np.zeros((sample_size))
    for index in range(sample_size):
        a[index] = c[index]    
    print(a)
    a = np.expand_dims(a, axis=0)
    b = np.zeros((a.shape[1], scale_range_size))
    a = a.astype(int)
    b = b.astype(int)
    for j in range(a.shape[1]):
        b[j,a[0,j]] += 1
    #b는 numpy array, 이걸 midi파일로 바꾸는 방법?
    b = np.transpose(b)
    b = np.flip(b, 0)
    bb = rnp.zeros((b.shape[0],b.shape[1]))     #cupy -> numpy 바꾸기
    for k in range(b.shape[0]):
        for j in range(b.shape[1]):
            bb[k,j] = b[k,j]
    plt.figure(i+1, figsize=(8,8))
    plt.imshow(bb)
    
