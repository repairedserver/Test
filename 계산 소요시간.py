import time
import numpy as np

arr = np.arange(9999999)
# 기본 연산
# sum = 0
# before = time.time()
# for i in arr:
#     sum += 1
# after = time.time()
# print(sum, after - before, "초")

# 벡터 연산
before = time.time()
sum = np.sum(arr)
after = time.time()
print(sum, after - before, "초")

#기본연산에 비해 벡터연산의 소요시간이 확연히 짧음