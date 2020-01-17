import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import data


# data里面的moon图像为单通道二维图像，故去掉第三个通道变量
def noisy(noise_type, image):
    if noise_type == "gauss":
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy

    elif noise_type == "s&p":
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0

        return out

    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_type == "speckle":
        row, col = image.shape
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = image + image * gauss
        return noisy


im = data.moon()
# 原始图像
plt.imshow(im, cmap='gray')
plt.title('Original Image')
plt.show()

# 椒盐噪声图像
sp_im = np.copy(im)
sp_im = noisy("s&p", sp_im)
plt.imshow(sp_im, cmap='gray')
plt.title('Salt&Pepper Noise Image')
plt.show()

# 高斯滤波图像
gauss_im = np.copy(im)
gauss_im = noisy("gauss", gauss_im)
plt.imshow(gauss_im, cmap='gray')
plt.title('Gauss Filter Image')
plt.show()

# 卷积核
kernel = np.ones((9, 9), np.float32) / 81

# kernel大小依次为3 5 7 9
kernel_sizes = list(range(3, 10, 2))
kernel_set = []
for kernel_size in kernel_sizes:
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    # ddepth为-1时目标图像和原图像卷积深度保持一致
    dst = cv.filter2D(sp_im, -1, kernel)
    plt.imshow(dst, cmap='gray')
    plt.title("Kernel with " + str(kernel_size) + " * " + str(kernel_size))
    plt.show()

# dst=cv.filter2D(sp_im,-1,kernel)

# laplacian=cv.Laplacian(dst,cv.CV_64F)

# dst2=dst-laplacian
# kLaplacian=np.array([[1,1,1],[1,-8,1],[1,1,1]])
# print(kLaplacian)
# laplacian2=cv.filter2D(dst.astype(np.float32),-1,kLaplacian)
# dst3=dst=laplacian2
