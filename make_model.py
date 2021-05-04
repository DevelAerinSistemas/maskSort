# -*- coding:utf-8 -*-
'''
  @ Author: Aerin Sistemas <aerin_proyectos@aerin.es>
  @ Create Time: 2021-05-04 11:07:27
  @ Modified time: 2021-05-04 11:07:40
  @ Project: AITea
  @ Description:
  @ License: MIT License
 '''



import cv2
import time
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


class MaskEstimate(object):
    def __init__(self, pos_folder, neg_folder, n_components=15, size=(69, 82)):
        self.pca = PCA(n_components)
        self.svm = SVC(kernel='linear', probability=True, random_state=42)
        self.size = size
        self.y_test = self.training(pos_folder, neg_folder)
        self.test_names = []

    def transform_image(self, img):
        img = img.resize(self.size)
        img = np.array(img)
        color_features = img.flatten()
        grey_img = rgb2gray(img)
        hog_features = hog(grey_img, block_norm='L2-Hys',
                           pixels_per_cell=(6, 6))
        return hog_features

    def get_images(self, folder, label=1, test=False):
        features = []
        y = []
        for dirpath, dnames, fnames in os.walk(folder):
            for file_name in fnames:
                if test:
                    self.test_names.append(fnames)
                    if "no" in file_name:
                        y.append(0)
                    else:
                        y.append(1)
                else:
                    y.append([label])
                img = Image.open(folder + file_name)
                hog_features = self.transform_image(img)
                features.append(hog_features)
        return y, np.stack(features, axis=0)

    def training(self, pos_folder, neg_folder):
        y_pos, X_pos, = self.get_images(folder=pos_folder, label=1, test=False)
        y_neg, X_neg = self.get_images(folder=neg_folder, label=0, test=False)
        y = y_pos + y_neg
        total_matrix = np.concatenate((X_pos, X_neg))
        pca_data_train = self.pca.fit_transform(total_matrix)
        self.svm.fit(pca_data_train, y)
        return y

    def test(self, test_folder):
        init_t = time.time()
        y_test,  X_test = self.get_images(
            folder=test_folder, label=0, test=True)
        X_test_pca = self.pca.transform(X_test)
        y_test_predict = self.svm.predict(X_test_pca)
        result = np.array(y_test_predict) - np.array(y_test)
        print(time.time() - init_t)
        return y_test_predict, y_test, sum(result == 0)/len(result)*100

    def distance(self, X):
        y = self.svm.decision_function(X)
        w_norm = np.linalg.norm(self.svm.coef_)
        dist = y / w_norm
        return dist

    def prediction(self, image):
        X = self.transform_image(image)
        X = self.pca.transform(X.reshape(1, -1))
        return self.distance(X), self.svm.predict(X)
