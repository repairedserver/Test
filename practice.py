import numpy as np

# 원소가 모두 3인 (3,4,5)형태의 numpy.array 출력
# arr = np.full((3, 4, 5), 3)
# print(arr)

# 정수 -50~50 범위 안의 난수로 이루어진 (4,5)형태의 
# numpy.array를 출력하고 행을 기준으로 오름차순 정렬과
# 전체배열을 1차원으로 변경해 오름차순 정렬결과 출력
arr = np.random.randint(-50, 50, (4, 5))
print(np.sort(arr, axis=0))
print(np.sort(arr, axis=None))