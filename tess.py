# import cv2
# import numpy as np

# # Load the image
# image = cv2.imread('E:/Pawang Code/Diabetic Retinopathy/dataset/2/3f5b4c2948e8.png')

# image = cv2.resize(image, (512, 512))

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply blur to reduce high frequency noise
# gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# # Apply Canny edge detection
# edged = cv2.Canny(gray_blurred, 10, 250)

# # Find contours
# contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# # Iterate over the contours
# for contour in contours:
#     # Check if the contour is small enough to be an exudate
#     if cv2.contourArea(contour) < 50:
#         continue
#     # Draw the contour on the image
#     cv2.drawContours(image, [contour], 0, (0, 0, 255), 2)

# # Show the image
# cv2.imshow('Exudate Detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Membaca gambar asli
image = cv2.imread('E:/Pawang Code/Diabetic Retinopathy/dataset/2/3f5b4c2948e8.png')

image = cv2.resize(image, (512, 512))

# Konversi gambar ke skala keabuan (grayscale)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplikasi operasi morfologi untuk menghilangkan noise dan menyempitkan pembuluh darah
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)

# Deteksi tepi menggunakan operator Canny
edges = cv2.Canny(gray, 50, 200, apertureSize=3)

# Menampilkan hasil deteksi tepi
cv2.imshow('Hasil Deteksi Tepi', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
