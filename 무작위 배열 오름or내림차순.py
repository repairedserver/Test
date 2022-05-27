import numpy as np
arr = np.random.randint(10, 20, (5, 5))
#오름차순
arr.sort(axis=1)
print(arr)

#내림차순
arr.sort(axis=1)
print(arr[:, ::-1])