# Segmentation_using_U_net

In 2018 kaggle have organized a Prediction Competition for creating an algorithm to automate nucleus detection.

Problem Statement: Find the nuclei in divergent images to advance medical discovery kaggle competition

I have tried here to build a U-NET architecture for semantic segmentation where i have choosen trainable parameter independent of the paper with the architecture remain same.

U-Net: Convolutional Networks for Biomedical Image Segmentation looks like:

![architecture](https://user-images.githubusercontent.com/51228517/140746084-73080b61-854a-4ce8-bff4-124a064566a8.PNG)

paper link: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

Dataset can be found using below link

dataset : https://www.kaggle.com/c/data-science-bowl-2018/data

Here are some segmentated result of the network:

Input image:

![x_test12](https://user-images.githubusercontent.com/51228517/140743747-9bfe0cd2-2067-4f94-82fd-4a342e50861c.jpg)

Segmented output:

![preds_test12](https://user-images.githubusercontent.com/51228517/140743792-67ffbe45-ea60-4ccc-bd04-30a59554d4d0.jpg)
