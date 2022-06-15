import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('Image.jpeg')

plt.imshow(img)
plt.show()