import imageio
import matplotlib.pyplot as plt
import numpy as np

im = imageio.imread('imageio:chelsea.png')


def LUT(image, lutTable):
    # image: grayscale or RGB color image
    # lutTable: [255,] 1D numpy array mapping 0-255 values to other values

    # lut = lambda x: lutTable[x]
    # return lut(image)

    # 不使用lambda函数
    # 这个也行
    # return lutTable[image]
    x = []
    for i in range(len(image)):
        x.append(lutTable[image[i]])
    return x


def GammaTable(gamma):
    invGamma = 1.0 / gamma
    #
    # table = np.array(
    #     [((i / 255.0) ** invGamma) * 255
    #         for i in np.arange(0, 256)]
    #                 ).astype("uint8")
    lst = []
    for i in np.arange(0, 256):
        lst.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(lst).astype("uint8")
    return table


R = im[:, :, 0]

gamma = 0.3
gamma = 1.5
R_crrt = LUT(R, GammaTable(gamma))

# 生成1行2列，这是第1个图
ax = plt.subplot(2, 2, 1)
plt.imshow(R, cmap='gray')
ax.set_title('Original image')

# 生成1行2列，这是第一个图
ax = plt.subplot(2, 2, 2)
plt.imshow(R_crrt, cmap='gray')
ax.set_title('gamma: ' + str(gamma))

# plt.show()

# 画曲线图
# plt.figure()
ax = plt.subplot(2, 2, 3)
X = np.array(range(0, 256))
plt.plot(X, GammaTable(1))
ax.set_title('gamma=1.0')
ax = plt.subplot(2, 2, 4)
plt.plot(X, GammaTable(gamma))
ax.set_title('gamma=' + str(gamma))
plt.show()
