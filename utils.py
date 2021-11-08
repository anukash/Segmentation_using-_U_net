# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:39:53 2021

@author: Anurag
"""

visualize = 0

import os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

img_width = 128
img_height = 128
img_channel = 3
def get_train_img_and_mask(train_path):
    train_id = next(os.walk(train_path))[1]

    x_train = np.zeros((len(train_id), img_width, img_height, img_channel), dtype=np.uint8)
    y_train = np.zeros((len(train_id), img_width, img_height, 1), dtype=np.bool)
    
    print('getting train image and mask ...')
    # print(next(os.walk('C:/Users/Anurag/Desktop/Study/segmentation_u_net/data/stage1_train/0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9/masks')))
    for n, ids in tqdm(enumerate(train_id), total=len(train_id)):
        path = train_path + '/'+ ids
        img = cv2.imread(path + '/images/' + ids + '.png')
        img = cv2.resize(img, (img_width, img_height))
        x_train[n] = img
        mask = np.zeros((img_width, img_height, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
            mask_ = np.expand_dims(cv2.resize(mask_, (img_width, img_height)), axis=-1)
            mask = np.maximum(mask, mask_)
        
        if visualize:
            if n ==5:
                break
        y_train[n] = mask
    print('Done!!!  got train images and mask')
    return x_train, y_train
    

def get_test(test_path): 
    print('getting test images....')
    test_id = next(os.walk(test_path))[1]
    x_test = np.zeros((len(test_id), img_width, img_height, img_channel), dtype=np.uint8)
    sizes_test = []
    for n, ids in tqdm(enumerate(test_id), total=len(test_id)):
        path = test_path + '/'+ ids
        img = cv2.imread(path + '/images/' + ids + '.png')
        sizes_test.append([img.shape[0], img.shape[1]])
        img = cv2.resize(img, (img_width, img_height))
        x_test[n] = img
    print('Done!!!  got test images')
    return x_test


if __name__ == '__main__':
    x_train, y_train = get_train_img_and_mask('data/stage1_train')
    plt.subplot(2,1,1)
    plt.imshow(x_train[3])
    plt.subplot(2,1,2)
    plt.imshow(np.squeeze(y_train[3]))
    plt.show()
    
    print(y_train[3].shape)
    print((np.squeeze(y_train[3])).shape)
    a = y_train[3]
    b = np.squeeze(y_train[3])