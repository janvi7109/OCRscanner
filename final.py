import cv2
import numpy as np
import pytesseract as pyt

from tkinter.filedialog import askopenfilename
filename = askopenfilename()

image = cv2.imread(filename)
image_work = np.copy(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

smooth = cv2.GaussianBlur(gray, (5,5), 0)

thresh = cv2.Canny(smooth, 100, 300)

contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if (len(contours)!=0):
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

epsilon = 0.05*cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

cv2.drawContours(image_work, [approx], -1, (255,0,0), 3)

cv2.waitKey(1000)

src = np.float32([approx[0,0], approx[1,0], approx[2,0], approx[3,0]])
dst = np.float32([[500,300],[500,0],[0,0],[0,300]])

M= cv2.getPerspectiveTransform(src,dst)
warped = cv2.warpPerspective(image, M,(500,300))

cv2.imshow('warped', warped)

cv2.waitKey(1000)

res = cv2.resize(warped, None, fx=8, fy=8, interpolation = cv2.INTER_CUBIC)
gray_warped = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
ret, thresh_warped = cv2.threshold(gray_warped,150,255, cv2.THRESH_BINARY_INV)




    
    

cv2.imshow("image", image_work)


cv2.destroyAllWindows()
text = pyt.image_to_string(thresh_warped)

print(text)

