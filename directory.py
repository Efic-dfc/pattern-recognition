# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 19:10:06 2021

@author: The Way
"""

# 公用目录
class Dirt:
    # 生成中间文件目录
    eigenface_path = 'data/eigenfaces/'
    # 原始图片目录
    rawdata_path = 'data/rawdata/'
    # 原始图片恢复成jpg格式后的存放目录
    restored_path = 'data/restored_data/'
    # 原始图片训练、测试集及其对应标签目录
    facelabel_path = 'data/facelabel/'
    # 存放LBP特征预处理结果之后的文件目录
    lbp_eigenface_path = eigenface_path + 'lbp/'
    # 存放LBP+PCA后特征降维结果之后的文件目录
    lbp_pca_eigenface_path = eigenface_path + 'lbp_pca/'
    # 输出测试结果图片的文件目录
    pic_path = 'image/'


# 整合全部图片，读成矩阵形式进行存放
ri_path = Dirt.eigenface_path + 'readImage'
# 数据索引目录
index_path = Dirt.eigenface_path + 'index'
# 原始数据训练集标签
faceDR_path = Dirt.facelabel_path + 'faceDR'
# 原始数据测试集标签
faceDS_path = Dirt.facelabel_path + 'faceDS'
# 全部标签整合后的csv文件路径
faceD_csv_path = Dirt.eigenface_path + 'faceD.csv'


class LBPFile:
    # 存放LBP特征预处理结果之后的文件目录
    _path = Dirt.lbp_eigenface_path
    # 经过LBP特征预处理后的训练集文件存放路径
    faceR_path = _path + 'faceR_lbp'
    # 经过LBP特征预处理后的测试集文件存放路径
    faceS_path = _path + 'faceS_lbp'
    # 经过LBP特征预处理后的训练集标签路径
    faceDR_path = _path + 'faceDR_lbp.csv'
    # 经过LBP特征预处理后的测试集标签路径
    faceDS_path = _path + 'faceDS_lbp.csv'


class LBP_PCAFile:
    # 存放LBP+PCA特征降维结果之后的文件目录
    _path = Dirt.lbp_pca_eigenface_path
    # 经过LBP+PCA特征降维后的训练集文件存放路径
    faceR_path = _path + 'faceR_lbp_pca'
    # 经过LBP+PCA特征降维后的测试集文件存放路径
    faceS_path = _path + 'faceS_lbp_pca'
    # 经过LBP+PCA特征降维后的训练集标签路径
    faceDR_path = _path + 'faceDR_lbp_pca.csv'
    # 经过LBP+PCA特征降维后的测试集标签路径
    faceDS_path = _path + 'faceDS_lbp_pca.csv'