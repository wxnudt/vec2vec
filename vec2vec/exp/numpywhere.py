#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# x = np.arange(12.).reshape(4, 3)
# print(x)
#
# y=np.where(x < 8, x, 1-x)
# print(y)


np.random.seed(0)
p = np.array([0.1, 0.4, 0.5, 0.1])
# print(p)
# pro=p/p.sum()
# print(pro)
for i in range(10):
    index = np.random.choice([0, 1, 2, 3],1, p = p/p.sum())
    print(index)