import cv2 as cv

img = cv.imread('Images/Cat face.jpg', 0)
img = cv.resize(img, (720, 720))

cv.imshow('Image', img)

cv.waitKey(0)
cv.destroyAllWindows()
