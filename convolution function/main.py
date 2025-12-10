import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


def conv2d_naive(image, kernel, stride=1, padding=0):



    if padding > 0:
        padded = np.zeros((image.shape[0] + 2*padding,
                           image.shape[1] + 2*padding), dtype=float)
        padded[padding:padding+image.shape[0],
               padding:padding+image.shape[1]] = image
    else:
        padded = image

    H, W = padded.shape
    KH, KW = kernel.shape


    out_h = (H - KH) // stride + 1
    out_w = (W - KW) // stride + 1
    output = np.zeros((out_h, out_w), dtype=float)


    for i in range(out_h):
        for j in range(out_w):
            region = padded[i*stride:i*stride+KH, j*stride:j*stride+KW]
            output[i, j] = np.sum(region * kernel)

    return output



if __name__ == "__main__":

    img_path = "./pct/mouse.png"  # 改成你自己的图片路径
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)



    print(f"size: {image.shape}")


    # kernel = np.array([
    #     [1, -1, -1],
    #     [1, 8, -1],
    #     [1, -1, -1]
    # ], dtype=float)
    kernel = np.array([
        [1, -1 ],
        [1, -1]
    ], dtype=float)


    start = time.time()
    feature_map = conv2d_naive(image, kernel, stride=1, padding=0)
    end = time.time()
    print(f"⏱️ 卷积完成，用时: {end - start:.4f} 秒")
    print(f"输出 feature map 尺寸: {feature_map.shape}")



    feature_map_norm = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX)
    feature_map_gray = np.uint8(feature_map_norm)


    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(feature_map_gray, cmap='gray')
    plt.title("2D Feature Map (Convolution Result)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


    cv2.imwrite("feature_map_output.png", feature_map_gray)
    print(" 已保存卷积结果到 feature_map_output.png")
