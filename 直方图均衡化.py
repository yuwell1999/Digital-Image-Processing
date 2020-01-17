import imageio
import matplotlib.pyplot as plt
import numpy as np

im = imageio.imread('imageio:chelsea.png')
# 把二维数组降维成一维
flat = im.flatten()
hist = plt.hist(flat, bins=255)
plt.show()


# create our own histogram function
def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)

    # loop through pixels and sum up counts of pixels
    for pixels in image:
        histogram[pixels] = histogram[pixels] + 1

    # return our final result
    return histogram


hist = get_histogram(flat, 256)

plt.plot(hist)
plt.show()


# create our cumulative sum function
def cumsum(a):
    rst = np.zeros(a.shape)
    rst[0] = a[0]
    # 积分
    for i in range(1, len(a)):
        rst[i] = rst[i - 1] + a[i]
    return rst


# exec the function
cs = cumsum(hist)

# display the result
plt.plot(cs)
plt.show()

# numerator & denomenator
# re-normalize cumsum values to be between 0-255
nj = (cs - cs.min()) * 255
N = cs.max() - cs.min()

cs = nj / N

plt.plot(cs)
plt.show()

# cast it back to uint8 since we can't use floating point values in images
cs = cs.astype('uint8')

plt.plot(cs)
plt.show()

# get the value from cumulative sum for every index in flat
img_new = cs[flat]

plt.hist(img_new, bins=50)
plt.show()

# put array back into original shape since we flattened it
img_new = np.reshape(img_new, im.shape)

ax = plt.subplot(121)
plt.imshow(im, cmap='gray')
ax.set_title('original image')

ax = plt.subplot(122)
plt.imshow(img_new, cmap='gray')
ax.set_title('hist equation')

plt.show()
