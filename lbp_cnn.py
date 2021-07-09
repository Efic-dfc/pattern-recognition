# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 17:17:42 2021

@author: The Way
"""

import algorithms.extra_feature_lbp as extra_feature_lbp
import algorithms.classify_cnn as classify_cnn
import time


startTime = time.time()

# 特征预处理方法：LBP
extra_feature_lbp.run(method_generateFaceRS='lbp')

# 分类：CNN
classify_cnn.run(method_readFaceRS='lbp')

endTime = time.time()

print('\nLBP_CNN costs %.2f seconds.' % (endTime - startTime))