#%matplotlib inline
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from scipy import misc #이미지 불러오기
import glob #특정 폴더의 파일 불러오기
import matplotlib.pyplot as plt 
import pickle as pkl
#안뇽 나는 규빈이라고 행

image_list1 = glob.glob("../resized/*.png") #이 부분 수정 가능성 있음
#parameters
the_number_of_files1 = len(image_list1)
channels = 1
width = 320
height = 180 
# from png to numpy_array you know what i'm saying?
np_images1 = np.zeros((the_number_of_files1, channels, width, height))
i = 0
for p in image_list1:
    image = misc.imread(p) #3차원 (세로크기,가로크기,채널수)
    image_reg = image[0:height, 0:width, 0:channels]/255.0
    np_image = np.array(image_reg)
    np_image = np.transpose(np_image, (2,1,0))
    np_image = np.expand_dims(np_image, axis=0)
    plt.figure(i+1)
    plt.imshow(np_image[0, 0, :, :])
    np_images1[i] = np_image
    i += 1
with open('pictures_for_encoder_input_train', 'wb') as f: 
    pkl.dump(np_images1, f)
print("train_pictures", np_images1.shape)
'''
plt.figure(i + 1, figsize = (4, 8))
plt.imshow(np.zeros((1,1)))
i += 1

image_list2 = glob.glob("../pictures_png/t*.png") #이 부분 수정 가능성 있음
#parameters
the_number_of_files2 = len(image_list2)
# from png to numpy_array you know what i'm saying?
np_images2 = np.zeros((the_number_of_files2, channels, width, height))
j=0
for t in image_list2:
    image = misc.imread(t) #3차원 (세로크기,가로크기,채널수)
    image_reg = image[0:height, 0:width, 0:channels]/255.0
    np_image = np.array(image_reg)
    np_image = np.transpose(np_image, (2,0,1))
    np_image = np.expand_dims(np_image, axis=0)
    plt.figure(i+1)
    plt.imshow(np_image[0, 0, :, :])
    np_images2[j] = np_image
    i += 1
    j += 1
with open('pictures_for_encoder_input_test', 'wb') as f: 
    pkl.dump(np_images2, f)
print("test_pictures", np_images2.shape)
'''
plt.figure(i + 1, figsize = (4, 8))
plt.imshow(np.zeros((1,1)))
i += 1

image_list3 = glob.glob("../resized/n*.png") #이 부분 수정 가능성 있음
#parameters
the_number_of_files3 = len(image_list3)
# from png to numpy_array you know what i'm saying?
np_images3 = np.zeros((the_number_of_files3, channels, width, height))
k=0
for t in image_list3:
    image = misc.imread(t) #3차원 (세로크기,가로크기,채널수)
    image_reg = image[0:height, 0:width, 0:channels]/255.0
    np_image = np.array(image_reg)
    np_image = np.transpose(np_image, (2,1,0))
    np_image = np.expand_dims(np_image, axis=0)
    plt.figure(i+1)
    plt.imshow(np_image[0, 0, :, :])
    np_images3[k] = np_image
    i += 1
    k += 1
with open('pictures_for_encoder_input_new', 'wb') as f: 
    pkl.dump(np_images3, f)
print("test_pictures", np_images3.shape)
