# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:59:49 2021

@author: Anurag
"""
from utils import get_test
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import cv2
test_path = 'data/stage1_test'

x_test = get_test(test_path)
print(x_test.shape)


model = keras.models.load_model('model_checkpoint/model.h5')
preds_test = model.predict(x_test, verbose=1)

preds_test_t = (preds_test > 0.5).astype(np.uint8)



for i in range(5):
    print(i)
    ix = random.randint(0, len(preds_test))
    plt.imshow(x_test[ix])
    plt.title(f'x_test{ix}')
    # plt.savefig(f'x_test{ix}')
    plt.show()
    plt.imshow(np.squeeze(preds_test[ix]))
    cv2.imshow('image',np.squeeze(preds_test[ix]))
    cv2.waitKey(0)
    plt.title(f'preds_test{ix}')
    # plt.savefig(f'preds_test{ix}')
    plt.show()

cv2.destroyAllWindows()