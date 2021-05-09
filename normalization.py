# -*- coding: utf-8 -*-

from sklearn import preprocessing
import numpy as np

a = np.random.random((1, 4))
print(a)
a = a*20
print("Data = ", a)

# normalize the data attributes
normalized = preprocessing.normalize(a)
print("Normalized Data = ", normalized)