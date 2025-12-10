import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def conv2d_naive(img, kernel, stride=1, padding=0):
    if padding > 0:
        tmp = np.zeros((img.shape[0] + 2*padding, img.shape[1] + 2*padding))
        tmp[padding:padding+img.shape[0], padding:padding+img.shape[1]] = img
        img = tmp

    h, w = img.shape
    kh, kw = kernel.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    out = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            part = img[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = np.sum(part * kernel)
    return out

if __name__ == "__main__":
    path = "./pct/mouse.png"  # 换成自己图片路径
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print("original_size:", img.shape)

    kernel = np.array([[1,-1,-1],[1,5,-1],[1,-1,-1]], float)
    # kernel = np.array([[1,-1],[1,-1]], float)

    t1 = time.time()
    out = conv2d_naive(img, kernel, stride=2, padding=5)
    t2 = time.time()

    print("done, time %.4fs" % (t2 - t1))
    print("size:", out.shape)

    norm = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    gray = np.uint8(norm)

    plt.figure(figsize=(9,4)) 
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    # plt.title("原图")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(gray, cmap='gray')
    # plt.title("卷积结果")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    cv2.imwrite("feature_map_output.png", gray)
    print("saving: feature_map_output.png")
