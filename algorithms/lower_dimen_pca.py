# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 23:22:03 2021

@author: The Way
"""

# coding=utf-8
from sklearn.decomposition import PCA
from fileio import generateUpdateFaceRS
from fileio import readFaceRS
import numpy as np


def lower_dimen(X_train, X_test, n_components):
    print('Start PCA lowering dimension...')

    X_train_index = X_train[:, 0].reshape(-1, 1)
    X_test_index = X_test[:, 0].reshape(-1, 1)

    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]

    pca = PCA(n_components=99)
    faceR = pca.fit_transform(X_train)
    faceS = pca.fit_transform(X_test)

    faceR = np.concatenate([X_train_index, faceR], axis=1)
    faceS = np.concatenate([X_test_index, faceS], axis=1)

    print('Shape of faceR: ' + str(faceR.shape))
    print('Shape of faceS: ' + str(faceS.shape))

    return faceR, faceS


def run(
    method_readFaceRS='densenet',
    method_generateUpdateFaceRS='densenet_pca',
    n_components=99
):
    '''
    特征提取+降维，第一个参数：特征提取方法，第二个参数：特征提取+降维方法
    '''
    # 读取已经经过特征处理过的四个文件作为输入
    X_train, X_test, y_train, y_test = readFaceRS(method_readFaceRS)

    # 特征提取+降维
    X_pca_train, X_pca_test = lower_dimen(X_train, X_test, n_components)

    # 把特征提取+降维的输出和标签输出
    generateUpdateFaceRS(
        X_pca_train, X_pca_test, y_train, y_test,
        method_generateUpdateFaceRS
)