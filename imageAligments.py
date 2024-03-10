import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("image1.jpg",0)
img2 = cv2.imread("Image2.jpg",0)

print("shape of images",img1.shape,img2.shape)

windowName = "original"
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)

windowNamenew = "Scanned"
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)

cv2.imshow(windowName,img1)
cv2.imshow(windowNamenew,img2)


## feature extractor, descriptor
MAX_NUM_FEATURES = 500
orb = cv2.ORB_create(MAX_NUM_FEATURES)

keypoint1, descriptor1 = orb.detectAndCompute(img1,None)
keypoint2, descriptor2 = orb.detectAndCompute(img2,None)

newImg1 = img1.copy()
newImg2 = img2.copy()

img_display1 = cv2.drawKeypoints(img1,keypoint1,outImage=np.array([]),color=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
img_display2 = cv2.drawKeypoints(img2,keypoint2,outImage=np.array([]),color=(0,255,0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)


windowName = "Image first"
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)

windowNamenew = "Image second"
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)

cv2.imshow(windowName,img_display1)
cv2.imshow(windowNamenew,img_display2)


# make a matcher
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

## return matches
mathces = matcher.match(descriptor1,descriptor2,None)

## take only good maches 
## Return a new list containing all items from the iterable in ascending order.
mathces = sorted(mathces,key= lambda x : x.distance,reverse=False)

## take only 10% of the matches

mathces = mathces[:int(len(mathces)*0.1)]
# print("Matches are ",mathces)

imageMatches = cv2.drawMatches(img1,keypoint1,img2,keypoint2,mathces,None)

cv2.imshow("Matches image",imageMatches)

## Work with homography

## points collect 
point1 = np.zeros((len(mathces),2), dtype=np.float32)
point2 = np.zeros((len(mathces),2), dtype=np.float32)

# print("shape of points are ",point1.shape)
# print("points are ",point1,point2)

# from matches we collect two things that is query index, train index

for i,match in enumerate(mathces):
    # print(point1[i,:])
    point1[i,:] = keypoint1[match.queryIdx].pt
    point2[i,:] = keypoint2[match.trainIdx].pt
    # print("query index is",match.queryIdx)
    # print("train index is", match.trainIdx)
  
# print("new points are ",point1,point2)

## lets find homography
h, mask = cv2.findHomography(point1,point2,cv2.RANSAC)

# print("Homography kernel size",h)
# print("homography mask is ",mask)

## we get h which is used for make wrapPerspective and convert out image 2 inthe same form of image 1

## Wrap image
height, width = img1.shape
print(img1.shape)

# img2New = cv2.warpPerspective(img2,h,(width,height))

# img2New = cv2.warpPerspective(img2,h,(width,height),flags=cv2.WARP_INVERSE_MAP)

# img2New = cv2.warpPerspective(img2,h,(width,height),flags=cv2.WARP_POLAR_LINEAR)

# img2New = cv2.warpPerspective(img2,h,(width,height),flags=cv2.WARP_POLAR_LOG)4

img2New = cv2.warpPerspective(img2,h,(width,height),flags=cv2.WARP_INVERSE_MAP)
## we get best outcome in WRAP_INVERSE_MAP 
## genrally we use WRAP_INVERSE_MAP only



## use appropriate flag to get best output 

cv2.imshow("new image ",img2New)
print(img2New.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()