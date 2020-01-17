import imageio
import matplotlib.pyplot as plt

# im = imageio.imread('C:/Users/YuYue/Desktop/IMG_6453.JPG')
im = imageio.imread('imageio:chelsea.png')
# 前两个向量全取，第三个向量选取一个通道
R1 = im[:, :, 0]
R2 = im[:, :, 1]
R3 = im[:, :, 2]


def ImThresh(im, minv, maxv):
    BinImg = im
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] > minv and im[i][j] < maxv:
                BinImg[i][j] = 1
            else:
                BinImg[i][j] = 0
    return BinImg


def ImThreshv2(im, minv, maxv):
    assert (len(im.shape) == 2)  #
    group1 = im >= minv
    group2 = im <= maxv
    return group1 * group2


# 生成1行2列，这是第1个图
ax = plt.subplot(1, 3, 1)
plt.imshow(R1)

# 生成1行3列，这是第2个图
ax = plt.subplot(1, 3, 2)
plt.imshow(R2)

# 生成1行3列，这是第3个图
ax = plt.subplot(1, 3, 3)
plt.imshow(R3)
plt.show()

im2 = ImThresh(R1, 1, 80)
plt.imshow(im2, cmap='gray')
plt.show()
