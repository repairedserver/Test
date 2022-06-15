import numpy as np
X = np.array([[51, 55], [14 ,19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])

X = X.flatten() # 1차원 배열로 변환
print(X)

print(X > 15)
print(X[X > 15])