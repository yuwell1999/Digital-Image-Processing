import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

im = np.zeros((512, 512), dtype=np.uint8)
# 中间100*100变成白色
im[200:300, 200:300] = 255
plt.imshow(im, cmap='gray')
plt.show()

# numpy库的fft模块的二维离散傅立叶变换函数fft2
# 输入一张灰度图，输出经过二维离散傅立叶变换后的结果
F = np.fft.fft2(im)
print("size of F:", F.shape)
print('Type of F’s value:', type(F[0, 0]))

#
S = np.abs(F)
plt.imshow(S, cmap='gray')
# 输出图像四个角有白色小点
plt.show()

#
Fs = np.fft.fftshift(F)
S = np.abs(Fs)
plt.imshow(S, cmap='gray')
# 输出图像中间有白色小点
plt.show()

S2 = np.log(S + 1)
plt.imshow(S2, cmap='gray')
# 输出图像出现十字形亮纹
plt.show()

print(S2.dtype)  # float 64

#
iFs = np.fft.ifftshift(Fs)
f_ = np.fft.ifft2(iFs)
im_ = np.abs(f_)

plt.imshow(im_, cmap='gray')
# 输出图像和原图非常接近
# 最大误差只有1.1368683772161603e-13
plt.show()

err = im_ - im
print(np.max(np.abs(err)))

K = np.ones((9, 9)) / 81.
im_blur = cv.filter2D(im, -1, K)
plt.imshow(im_blur, cmap='gray')
plt.show()
# 图像利用卷积被模糊了，差别最大为228
err = im_blur - im
print(np.max(np.abs(err)))

K_pad = np.zeros(im.shape)
#
offset = int((im.shape[0] - K.shape[0]) / 2)
#
K_pad[offset:offset + K.shape[0], offset:offset + K.shape[0]] = K

#
F_K = np.fft.fft2(K_pad)
#
D = F_K * F
tmp = np.fft.ifft2(D)
im_D = np.real(tmp)
plt.imshow(im_D, cmap='gray')
# 四个角有较大的模糊白块
plt.show()

K_pad = np.zeros(im.shape)
offset = int((im.shape[0] - K.shape[0]) / 2)
K_pad[offset:offset + K.shape[0], offset:offset + K.shape[0]] = K
F_K = np.fft.fft2(K_pad)
D = F_K * F
tmp = np.fft.fftshift(np.fft.ifft2(D))
im_D = np.real(tmp)
plt.imshow(im_D, cmap='gray')
# 模糊白块回到中心，位置稍有变化
plt.show()

plt.imshow(np.abs(F_K))
# 图像变成紫色，四个角偏绿色
plt.show()

kLap = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
kLap_pad = np.zeros(im.shape)
offset = int((im.shape[0] - kLap.shape[0]) / 2)
kLap_pad[offset:offset + kLap.shape[0], offset:offset + kLap.shape[0]] = kLap

KLap = np.fft.fft2(kLap_pad)
D_Lap = KLap * F
tmp = np.fft.ifftshift(np.fft.ifft2(D_Lap))
im_D_Lap = np.real(tmp)

plt.imshow(im_D_Lap, cmap='gray')
plt.show()

# use ordinary spatial filtering method
imLap = cv.filter2D(im.astype(np.float32), -1, kLap)
plt.figure()
plt.imshow(imLap, cmap='gray')
#
plt.show()

plt.imshow(np.abs(KLap))
plt.show()


def FreqFilter(grayIm, kernel, bShowFreq=False):
    # padding the kernal to enlarge it to the same size with the image
    k_pad = np.zeros(grayIm.shape)
    offset0 = int((grayIm.shape[0] - kernel.shape[0]) / 2.)
    offset1 = int((grayIm.shape[1] - kernel.shape[1]) / 2.)
    k_pad[offset0:offset0 + kernel.shape[0], offset1:offset1 + kernel.shape[1]] = kernel

    FK = np.fft.fft2(k_pad)
    FI = np.fft.fft2(grayIm)

    FKs = np.fft.fftshift(FK)
    FIs = np.fft.fftshift(FI)

    dotMult = FI * FK

    result = np.fft.ifft2(dotMult)

    if bShowFreq:
        plt.subplot(121)
        plt.title('freq4Image')
        plt.imshow(np.abs(FIs))

        plt.subplot(122)
        plt.title('freq4Kernel')
        plt.imshow(np.abs(FKs))

    return np.real(np.fft.fftshift(result))


filtered = FreqFilter(im, kLap, True)

plt.imshow(filtered, cmap='gray')
plt.show()

kEdge = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
edgIm = cv.filter2D(im.astype(np.float32), -1, kEdge)
plt.imshow(edgIm, cmap='gray')
plt.show()

filtered = FreqFilter(im, kEdge, True)
plt.imshow(filtered, cmap='gray')
plt.show()

# 1、为什么频率滤结束后要添加fftshift才能看到正确的滤波结果
'''
图像中的高频信号代表的是图像中的细节和边缘信息。
而图像中的大部分内容信息，往往是低频信号，
所以，信号中的低频信号代表更多的图像信息。
在iff2操作后，得到的频率域图像是低频在四周，高频在中间区域。
为了还原图像，进行fftshift操作，将低频分量移动到频谱的中央。
'''

# 2、频率滤波结果与基于opencv的空间滤波有何区别
