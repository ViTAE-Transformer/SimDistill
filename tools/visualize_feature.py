import os
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config
import glob
import torch
import io
from torchvision import transforms
from torch.utils.data import DataLoader
import PIL.Image as pil
from PIL import Image
#import imageio
import scipy.ndimage
def mkdir(path):

    isExists = os.path.exists(path) # 判断路径是否存在，若存在则返回True，若不存在则返回False
    if not isExists: # 如果不存在则创建目录
        os.makedirs(path)
        return True
    else:
        return False

def save_feature_to_img_cam(features, idx, visual_path):
    visual_path = visual_path.split("LIDAR_TOP/")[1].split(".pcd.bin")[0]
    # features=self.get_single_feature() # 返回一个指定层输出的特征图,属于四维张量[batch,channel,width,height]
    for i in range(features.shape[1]):
        print("----------", features.shape)
        feature = features[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量

        feature = feature.cpu().data.numpy() # 转为numpy

        # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        # img = cv2.resize(img, (256, 256))
        print("type: ", type(feature))
        print("shape: ", feature.shape)
        #feature = cv2.resize(feature, (256, 256))
        feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！
        feature = np.uint8(255 * feature)#np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行
        feature = cv2.resize(feature, (256, 256))
        heatmap = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
        path = os.path.join("outfeature/Cam/", visual_path)#   visaul_path+idx
        mkdir(path)  # 创建保存文件夹，以选定可视化层的序号命名
        print(path)
        superimposed_img = heatmap * 0.4 #+ img
        cv2.imwrite(path +'/' +str(i) + '.jpg',superimposed_img)  # 保存当前层输出的每个channel上的特征图为一张图像
        #cv2.imwrite(path  + str(i) + '.jpg', superimposed_img)


def save_feature_to_img_lidarbranch(features, idx, visual_path):
    visual_path = visual_path.split("LIDAR_TOP/")[1].split(".pcd.bin")[0]
    # features=self.get_single_feature() # 返回一个指定层输出的特征图,属于四维张量[batch,channel,width,height]
    for i in range(features.shape[1]):
        print("----------", features.shape)
        feature = features[:, i, :, :]  # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        feature = feature.view(feature.shape[1], feature.shape[2])  # batch为1，所以可以直接view成二维张量

        feature = feature.cpu().data.numpy()  # 转为numpy

        # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        # img = cv2.resize(img, (256, 256))
        #feature = cv2.resize(feature, (256, 256))
        feature = (feature - np.amin(feature)) / (np.amax(feature) - np.amin(feature) + 1e-5)  # 注意要防止分母为0！
        feature = np.uint8(255 * feature)  # np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行
        feature = cv2.resize(feature, (256, 256))
        heatmap = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
        #path = os.path.join(visual_path, str(idx))  # visaul_path+idx
        path = os.path.join("outfeature/Plidar", visual_path)
        mkdir(path)  # 创建保存文件夹，以选定可视化层的序号命名
        # print(path)
        superimposed_img = heatmap * 0.4  # + img
        cv2.imwrite(path +'/' +str(i) + '.jpg',superimposed_img)  # 保存当前层输出的每个channel上的特征图为一张图像
        #cv2.imwrite("outfeature/" + path + '/Plidarbranch/' + str(i) + '.jpg', superimposed_img)
