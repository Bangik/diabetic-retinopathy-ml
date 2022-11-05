from time import time
import cv2
import numpy as np
from pyparsing import col
import skimage.morphology as morph
import skimage.filters as filters
import skimage.exposure as exposure
import xlsxwriter as xls
from skimage.feature import greycomatrix, greycoprops
import os

path_dataset = "D:/Pawang Code/Diabetic Retinopathy/dataset"
book = xls.Workbook('D:/Pawang Code/Diabetic Retinopathy/featureExtraction1.xlsx')
sheet = book.add_worksheet()
sheet.write(0, 0, 'Image')
column = 1
glcm_feature = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
angle = [0]
for i in range(len(glcm_feature)):
    for j in range(len(angle)):
        sheet.write(0, column, glcm_feature[i] + '_' + str(angle[j]))
        column += 1
sheet.write(0,7, 'Label')
column += 1
row = 1

def preprocessing(img):
    img = cv2.resize(img, (500, 500)) # Resize image
    green = img[:, :, 1] # Get green channel
    incomplement = cv2.bitwise_not(green) # negative image
    clache = cv2.createCLAHE(clipLimit=5) # Contrast Limited Adaptive Histogram Equalization
    cl1 = clache.apply(incomplement) # Apply CLAHE
    mopopen = morph.opening(cl1, morph.disk(8, dtype=np.uint8)) # Morphological opening with disk kernel of radius 8
    godisk = cl1 - mopopen #remove optical disk
    medfilt = filters.median(godisk) # Median filter
    background = morph.opening(medfilt, morph.disk(15, dtype=np.uint8)) #get background
    rmBack = medfilt - background #remove background
    v_min, v_max = np.percentile(rmBack, (0.2, 99.8)) #get 0.2% and 99.8% percentile
    better_contrast = exposure.rescale_intensity(rmBack, in_range=(v_min, v_max)) #rescale intensity
    ret, thresh = cv2.threshold(better_contrast, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Otsu thresholding
    rmSmall = morph.remove_small_objects(thresh, min_size=50, connectivity=1, in_place=False) #remove small objects

    return rmSmall

def glcm(img):
    distance = [5]
    angles = [0]
    level = 256
    symetric = True
    normed = True

    glcm = greycomatrix(img, distance, angles, level, symmetric=symetric, normed=normed)
    glcm_props = [property for name in glcm_feature for property in greycoprops(glcm, name)[0]]

    return glcm_props

for folder in os.listdir(path_dataset):
    sub_folder_files = os.listdir(os.path.join(path_dataset, folder))
    len_sub_folder = len(sub_folder_files)
    for i, filename in enumerate(sub_folder_files):
        column = 0
        print('Processing image {}/{} in folder {}'.format(i+1, len_sub_folder, folder))
        sheet.write(row, column, filename)
        column += 1

        img = cv2.imread(os.path.join(path_dataset, folder, filename))
        preprocessed = preprocessing(img)

        glcm_props = glcm(preprocessed)
        for prop in glcm_props:
            sheet.write(row, column, prop)
            column += 1
        
        sheet.write(row, column, folder)
        column += 1

        row += 1
        if i == 99:
            break

book.close()