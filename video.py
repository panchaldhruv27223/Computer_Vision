import cv2
import matplotlib.pyplot as plt
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")
outputimg = cv2.VideoWriter("outVideo.mp4",fourcc,30,(720,720),True)

cap = cv2.VideoCapture(0)

while 1 :
    hasFrame, frame = cap.read()
    
    if hasFrame:
        
        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame,(720,720),interpolation=cv2.INTER_CUBIC)
        outputimg.write(frame)
        
        cv2.imshow("myVideo",frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
cv2.destroyAllWindows()
cap.release()
outputimg.release()