import cv2 
import matplotlib
import numpy as np


images = []

for i in range(1,10):
    images.append(cv2.imread(f"im{i}.jpg"))

sticher = cv2.Stitcher_create()
status, result = sticher.stitch(images)
print("the status is ",status)


window = "my image"
cv2.namedWindow(window,cv2.WINDOW_NORMAL)
cv2.imshow(window,result)



cv2.waitKey(0)
cv2.destroyAllWindows()