import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern


'''
    Realize a leitura das imagens de tomografias.
    Aplique qualquer técnica de pré-processamento necessária (normalização, equalização de histograma, etc.)
'''
def processImages(directory):
    images = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                label = os.path.basename(root)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (128, 128))  
                images.append(img)
                labels.append(label)
    return images, labels

images, labels = processImages("brain-tumor-mri-dataset")

'''
 Extraia as características das imagens utilizando HOG.
'''
def hog(image):
    hog = cv2.HOGDescriptor()
    hog_caract = hog.compute(image)
    return hog_caract.flatten()

hog_caract = [hog(img) for img in images]

'''
    Extração de histograma das imagens 
'''
def histograma(image):
    hist = cv2.calcHist([image],[0], None, [256], [0,256])
    cv2.normalize(hist, None, 0, 1.0, cv2.NORM_MINMAX)
    return hist

histograma_caract = [histograma(img) for img in images]


'''
    Extração característica LBP de todas as imagens
'''

def lbp(imagem):
    raio = 3
    pontos = 8 * raio
    lbp = local_binary_pattern(imagem, pontos, raio, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, pontos + 3), range=(0, pontos + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7) #normalização do histograma 
    return hist


lbp_caract = [lbp(img) for img in images]

