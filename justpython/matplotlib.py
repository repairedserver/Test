import numpy as np
import matplotlib.pyplot as plt

#데이터준비
x = np.arange(0, 6, 0.1) # 0~6까지 0.1 간격으로 생성
y = np.sin(x)

plt.plot(x, y)
plt.show()