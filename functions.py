import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature


def read_images(directory: str) -> tuple:
    images = list()
    labels = list()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                print(f"Reading {img_path}")
                label = os.path.basename(root)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
    return images, labels


def process_images(image):
    hist1 = cv2.calcHist([image], [0], None, [256], [0, 256])
    image = cv2.equalizeHist(image)
    hist2 = cv2.calcHist([image], [0], None, [256], [0, 256])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    hist3 = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, None, 0, 1.0, cv2.NORM_MINMAX)
    hist2 = cv2.normalize(hist2, None, 0, 1.0, cv2.NORM_MINMAX)
    hist3 = cv2.normalize(hist3, None, 0, 1.0, cv2.NORM_MINMAX)
    # fig, axs = plt.subplots(3)
    # fig.suptitle('Histogram')
    # axs[0].plot(hist1)
    # axs[1].plot(hist2)
    # axs[2].plot(hist3)
    # plt.show()
    return image


def apply_filter(image):
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    b_image = cv2.GaussianBlur(image, (7, 7), 0)
    (_, t_image) = cv2.threshold(b_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    image = cv2.bitwise_and(image, image, t_image)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image


def extraction_hog(image):
    hog = cv2.HOGDescriptor()
    hog_characteristic = hog.compute(image)
    return hog_characteristic


def extraction_lbp(image):
    lbp = feature.local_binary_pattern(image, 8, 1, method='uniform')
    (lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    return lbp
