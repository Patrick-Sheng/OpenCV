import cv2

img = cv2.imread('Images/Cat face.jpg', 1)


# image, "Window name", "org, co-ordinate", font, fontScale, colour, thickness, cv2.LINE_AA
image = cv2.putText(img, 'OpenCV', (100, 50), cv2.FONT_ITALIC, 1, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
