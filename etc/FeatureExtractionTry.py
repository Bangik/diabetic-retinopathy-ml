from time import time
import cv2
import numpy as np
from pyparsing import col
import skimage.morphology as morph
import skimage.filters as filters
import skimage.exposure as exposure
import xlsxwriter as xls
from skimage.feature import graycomatrix, graycoprops
import skimage.feature as feature
import os

path_dataset = "E:/Pawang Code/Diabetic Retinopathy/dataset"
book = xls.Workbook('E:/Pawang Code/Diabetic Retinopathy/featureExtractionbv.xlsx')
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

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img>tol        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
            return img

def circle_crop_v2(img):
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img

def preprocessing(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, mg = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    # reverse = cv2.bitwise_not(mg)
    # contours, hierarchy = cv2.findContours(reverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # select = max(contours, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(select)
    # png = img[y:y+h, x:x+w]
    # png = cv2.resize(png, (500, 500))
    # green = png[:, :, 1]
    # clahe = exposure.equalize_adapthist(green, clip_limit=0.03)
    green = img[:,:,1]
    incomplement = cv2.bitwise_not(green) # negative image
    clache = cv2.createCLAHE(clipLimit=5) # Contrast Limited Adaptive Histogram Equalization
    cl1 = clache.apply(incomplement) # Apply CLAHE

    return cl1

def extract_bv(image):
    retina = image[:, :, 1]
    t0, t1 = filters.threshold_multiotsu(retina, classes=3)
    mask = (retina > t0)
    vessels = filters.sato(retina, sigmas=range(1, 10)) * mask
    thresholded = filters.apply_hysteresis_threshold(vessels, 0.01, 0.03)
    return thresholded

def glcm(img):
    distance = [5]
    angles = [0]
    level = 256
    symetric = True
    normed = True

    glcm = graycomatrix(img, distance, angles, level, symmetric=symetric, normed=normed)
    glcm_props = [property for name in glcm_feature for property in graycoprops(glcm, name)[0]]

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
        img = circle_crop_v2(img)
        preprocessed = extract_bv(img)

        glcm_props = glcm(preprocessed)
        for prop in glcm_props:
            sheet.write(row, column, prop)
            column += 1
        
        sheet.write(row, column, folder)
        column += 1

        row += 1
        # if i == 99:
        #     break

book.close()