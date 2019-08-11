# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:18:10 2019

@author: kaswa
"""

from os import listdir, makedirs
import sys
from os.path import exists, join
from PIL import Image #for image resizing

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])#, ".jpg", ".jpeg", "gif"])
'''
if len(sys.argv) == 1:
    print("Error! No folder Name")
    print("usage: resize_image.py folder_name")
    sys.exit()
'''
#image_dir = sys.argv[1]
image_dir = 'C:\Users\kebinxyz\Documents\GitHub\PicturesToMelodies\pictures_png'
print(image_dir)

file_list = listdir(image_dir)
print(file_list)

image_file_list = [x for x in listdir(image_dir) if is_image_file(x)]
print(image_file_list)


if not exists("resized"): 
    makedirs("resized")
    
for image_file_name in image_file_list:
    im = Image.open(join(image_dir, image_file_name))
    image2 = im.resize((256, 256))

    save_path = join('resized/', image_file_name)
    print (save_path)
    image2.save(save_path)
          