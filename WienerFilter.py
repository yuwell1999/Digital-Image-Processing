import math

import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from skimage import color
from skimage import io

im = color.rgb2gray(io.imread('C:/Users/YuYue/Desktop/timg.jfif'))

im_height = im.shape[0]
im_width = im.shape[1]
plt.imshow(im, cmap='gray')
plt.title("Original Image")
plt.show()


# simulate a motion blur
def motion_blur(image_size, motion_length, motion_angle):
    PSF = np.zeros(image_size)
    center_position = (image_size[0] - 1) / 2
    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(motion_length):
            offset = round(i * slope_tan)
            PSF[int(center_position + offset), int(center_position - offset)] = 1
        return PSF / PSF.sum()
    else:
        for i in range(motion_length):
            offset = round(i * slope_cot)
            PSF[int(center_position - offset), int(center_position + offset)] = 1
        return PSF / PSF.sum()


def make_blurred(input, PSF, eps):
    input_fft = np.fft.fft2(input)  # 进行二维数组的傅里叶变换
    PSF_fft = np.fft.fft2(PSF) + eps
    blurred = np.fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(np.fft.fftshift(blurred))
    return blurred


PSF = motion_blur((im_height, im_width), 20, 45)
im_motion_blur = np.abs(make_blurred(im, PSF, 0.01))

plt.imshow(im_motion_blur, cmap='gray')
plt.title("Motion Blur")
plt.show()

# simulate additive noise
'''
im_gaussian=cv.GaussianBlur(im_motion_blur,ksize=(15,15),sigmaX=0,sigmaY=0)
plt.imshow(im_gaussian,cmap='gray')
plt.title("Motion Blur with Gauss Noise")
plt.show()
'''

im_gaussian = im_motion_blur + 0.1 * im_motion_blur.std() * np.random.standard_normal(im_motion_blur.shape)

plt.imshow(im_gaussian, cmap='gray')
plt.title("Motion Blur with Gauss Noise")
plt.show()


# try restoration assuming no noise
def inverse(im, PSF, eps):  # 逆滤波
    im_fft = fft.fft2(im)
    PSF_fft = fft.fft2(PSF) + eps  # 噪声功率，这是已知的，考虑epsilon
    result = fft.ifft2(im_fft / PSF_fft)  # 计算F(u,v)的傅里叶反变换
    result = np.abs(fft.fftshift(result))
    return result


def wiener(im, PSF, eps, K=0.01):  # 维纳滤波，K=0.01
    im_fft = fft.fft2(im)
    PSF_fft = fft.fft2(PSF) + eps
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = fft.ifft2(im_fft * PSF_fft_1)
    result = np.abs(fft.fftshift(result))
    return result


im_inv = wiener(im_gaussian, PSF, 0.0001, 4)
plt.imshow(im_inv, cmap='gray')
plt.title("Inverse Filtering")
plt.show()

# try restoration using a better estimate of the noise-to-signal-power ratio
im_wiener = wiener(im_gaussian, PSF, 0.001, 0.01)
plt.imshow(im_wiener, cmap='gray')
plt.title("Wiener Filtering")
plt.show()
