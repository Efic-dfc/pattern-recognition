# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 20:41:47 2021

@author: The Way
"""

# coding=utf-8
from sklearn.model_selection import train_test_split
import directory as dirt
from cv2 import cv2
import pandas as pd
import numpy as np
import os
import re


def mkdir_all(datapath):
    print('Generating paths: ')
    for path in datapath:
        print(path)
        if not os.path.exists(path):
            os.mkdir(path)


def generateRi():
    '''
    全部图片读成矩阵形式存放
    '''
    raw_files = os.listdir(dirt.Dirt.rawdata_path)

    times = 0

    for raw_file in raw_files:
        with open(dirt.Dirt.rawdata_path + raw_file) as f:
            In = np.fromfile(f, dtype=np.uint8).reshape(1, -1)

            if times == 0:
                ri = In
            else:
                if ri.shape[1] != In.shape[1]:
                    In = cv2.resize(In, (ri.shape[1], 1))
                ri = np.concatenate((In, ri), axis=0)

            times += 1

            print('ri-shape:' + str(ri.shape))

    np.savetxt(dirt.ri_path, ri, fmt='%d')

    print('ri is done:' + str(ri.shape))


def generateIndex():
    '''
    生成图片的编号索引，并存入index文件中
    '''
    raw_files = os.listdir(dirt.Dirt.rawdata_path)
    raw_files = list(map(int, raw_files))
    raw_files = np.array(raw_files, dtype='int32')
    np.savetxt(dirt.index_path, raw_files, fmt='%d')


def readRiIndex():
    '''
    读取所有图片的大矩阵中矩阵的维度和索引编号
    '''
    ri = np.loadtxt(dirt.ri_path)
    print('ri import done. ri-shape: ' + str(ri.shape))

    index = np.loadtxt(dirt.index_path, dtype='int32').reshape(-1, 1)
    print('index import done. ri-shape: ' + str(index.shape))

    return ri, index


def generateCSV():
    '''
    原始的全部标签文本文件转换为CSV格式
    '''
    result = pd.DataFrame(columns=('idx', 'sex', 'age', 'race', 'face', 'prop'))

    with open(dirt.faceDR_path) as f:
        lines = f.readlines()

    for line in lines:
        if not re.search('_missing', line):
            index = re.match(' ([0-9]{4})', line).group(1)
            print(index)
            sex = re.search('\(_sex  (.*?)\)', line).group(1)
            age = re.search('\(_age  (.*?)\)', line).group(1)
            race = re.search('\(_race (.*?)\)', line).group(1)
            face = re.search('\(_face (.*?)\)', line).group(1)
            prop = re.search('\(_prop \'\((.*?)\)', line).group(1)
            if not prop:
                prop = 'None'
            result = result.append(pd.DataFrame({'idx': [index], 'sex': [sex], 'age': [
                                   age], 'race': [race], 'face': [face], 'prop': [prop]}), ignore_index=True)
        else:
            continue

    with open(dirt.faceDS_path) as f:
        lines = f.readlines()
    for line in lines:
        if not re.search('_missing', line):
            index = re.match(' ([0-9]{4})', line).group(1)
            print(index)
            sex = re.search('\(_sex  (.*?)\)', line).group(1)
            age = re.search('\(_age  (.*?)\)', line).group(1)
            race = re.search('\(_race (.*?)\)', line).group(1)
            face = re.search('\(_face (.*?)\)', line).group(1)
            prop = re.search('\(_prop \'\((.*?)\)', line).group(1)
            if not prop:
                prop = 'None'
            result = result.append(pd.DataFrame({'idx': [index], 'sex': [sex], 'age': [
                                   age], 'race': [race], 'face': [face], 'prop': [prop]}), ignore_index=True)
        else:
            continue

    result.to_csv(dirt.faceD_csv_path, index=False)


def generateFaceRS(X, method):
    '''
    特征处理的脚本中执行，进行特征处理，划分数据集，将结果写到方法所在的目录里
    '''
    y = pd.read_csv(dirt.faceD_csv_path)

    print('X-shape: ' + str(X.shape))
    print('y-shape: ' + str(y.shape))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7
    )

    if (method == 'lbp'):
        file_path = dirt.LBPFile

    np.savetxt(
        file_path.faceR_path, X_train,
        fmt=['%d']+['%.6f']*(X_train.shape[1]-1)
    )
    np.savetxt(
        file_path.faceS_path, X_test,
        fmt=['%d']+['%.6f']*(X_test.shape[1]-1)
    )

    y_train.to_csv(file_path.faceDR_path, index=False)
    y_test.to_csv(file_path.faceDS_path, index=False)

    print('write file to ' + file_path._path)
    print('[write] X_train.shape, X_test.shape, y_train.shape, y_test.shape:')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def readFaceRS(method):
    '''
    用到特征处理结果的时候，执行该函数，读取特征处理后的训练、测试集结果，便于进行下一步降维/分类
    method - 上一步使用方法
    '''
    if (method == 'lbp'):
        file_path = dirt.LBPFile
    elif (method == 'lbp_pca'):
        file_path = dirt.LBP_PCAFile

    # 读取数据
    X_train = np.loadtxt(file_path.faceR_path)
    X_test = np.loadtxt(file_path.faceS_path)

    y_train = pd.read_csv(file_path.faceDR_path, index_col=0)
    y_test = pd.read_csv(file_path.faceDS_path, index_col=0)

    print('read file from ' + file_path._path)
    print('[read] X_train.shape, X_test.shape, y_train.shape, y_test.shape:')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


def generateUpdateFaceRS(X_train, X_test, y_train, y_test, method):
    '''
    两种特征提取方法后的输出
    '''
    if (method == 'lbp_pca'):
        file_path = dirt.LBP_PCAFile

    np.savetxt(
        file_path.faceR_path, X_train,
        fmt=['%d']+['%.6f']*(X_train.shape[1]-1)
    )
    np.savetxt(
        file_path.faceS_path, X_test,
        fmt=['%d']+['%.6f']*(X_test.shape[1]-1)
    )

    y_train.to_csv(file_path.faceDR_path)
    y_test.to_csv(file_path.faceDS_path)

    print('write file to ' + file_path._path)
    print('[write] X_train.shape, X_test.shape, y_train.shape, y_test.shape:')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


if __name__ == '__main__':
    directories_dict = dirt.Dirt.__dict__
    directories = [directories_dict[key]
                   for key in directories_dict if "__" not in key]

    mkdir_all(directories)

    if not os.path.exists(dirt.ri_path):
        generateRi()

    if not os.path.exists(dirt.index_path):
        generateIndex()

    if not os.path.exists(dirt.faceD_csv_path):
        generateCSV()