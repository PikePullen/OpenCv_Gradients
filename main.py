import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_image():
    blank_img = np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, text='ABCDE', org=(50,300), fontFace=font, fontScale=5, color=(255,255,255), thickness=25)
    return blank_img

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

img = cv2.imread('../DATA/sudoku.jpg',0)
display_img(img)

""" Sobel operators """
# depth is precision of pixel
# vertical lines are very clear
sobelx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
display_img(sobelx)

# horizontal lines are very clear
sobely = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
display_img(sobely)

# attempts to gradient along both axis
laplacian = cv2.Laplacian(img, cv2.CV_64F)
display_img(laplacian)

# blended view of two types
blended = cv2.addWeighted(src1=sobelx, alpha=0.5, src2=sobely, beta=0.5, gamma=0)
display_img(blended)

# displays numbers clearly
ret,th1 = cv2.threshold(img, 100, 255,cv2.THRESH_BINARY)
display_img(th1)

# combining previous techniques
kernel = np.ones((4,4), np.uint8)
gradient = cv2.morphologyEx(blended, cv2.MORPH_GRADIENT, kernel)
display_img(gradient)