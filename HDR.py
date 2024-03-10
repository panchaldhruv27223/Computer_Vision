import cv2
import numpy as np 
import matplotlib.pyplot as plt

img1 = cv2.imread("images1.png")
img2 = cv2.imread("images2.png")
img3 = cv2.imread("images3.png")
img1 = cv2.resize(img1,(720,720),img1,interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(img2,(720,720),img2,interpolation=cv2.INTER_CUBIC)
img3 = cv2.resize(img3,(720,720),img3,interpolation=cv2.INTER_CUBIC)

# window = "image show"
# cv2.namedWindow(window,cv2.WINDOW_NORMAL)

# cv2.imshow(window,img1)
# cv2.imshow(window,img2)
# cv2.imshow(window,img3)

images = [img1,img2,img3]
times = np.array([1/30,0.25,2.5],dtype=np.float32)


## image Aligments
aligMTB = cv2.createAlignMTB()
aligMTB.process(images,images)

## Create a Camera Response Function

## calibrate debe vec method - 1
calibrateDebeVec = cv2.createCalibrateDebevec()
responseDebeVec = calibrateDebeVec.process(images,times)
## it will give inverse camera response function
# plot it and see it 
x = np.arange(256,dtype=np.uint8)
# print(x.shape)
y = np.squeeze(responseDebeVec)

# print(y)
# print(y.shape)
# plt.plot(x,y[:,0],"b",x,y[:,1],"g",x,y[:,2],"r")
# plt.show()

#  
window = "image show"
cv2.namedWindow(window,cv2.WINDOW_NORMAL)

cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("img3",img3)


## Merge Exposure into an HDR Image
mergerDebevec = cv2.createMergeDebevec()
hdrdebevec = mergerDebevec.process(images,times,responseDebeVec)


# Tonemapping
## take 24 bit color image and convert it into 8 bit color image 
tonemapDrago = cv2.createTonemapDrago(1.0,1.0)
ldrdrago = tonemapDrago.process(hdrdebevec)
ldrdrago = ldrdrago*3
# ldrdrago = np.clip(ldrdrago, 0, 1)
cv2.imshow("new img",255*ldrdrago[:,:,::-1])

cv2.waitKey(0)
cv2.destroyAllWindows()