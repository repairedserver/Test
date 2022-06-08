import time
import numpy as np

arr1 = np.arange(9999999)
arr2 = np.arange(9999999)
sum = 0

#기본 연산
# before = time.time()
# for i, j in zip(arr1, arr2):
#     sum += i * j
# after = time.time()
# print(sum, after - before, "초")

#벡터 연산
before = time.time()
sum = np.dot(arr1, arr2)
after = time.time()
print(sum, after - before, "초")

#기본 연산에 비해 벡터 연산이 더 빠름