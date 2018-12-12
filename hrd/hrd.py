import numpy as np
import cv2

img = cv2.imread("../dhokHussu.jpg")
w,h,c = img.shape

img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(40, 40))
c_img = clahe.apply(img_gray)
cv2.imshow("c_img",c_img)
cv2.imwrite("../c_img.jpg",c_img)
# thr = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,3)
ret,thr = cv2.threshold(c_img,50,255,cv2.THRESH_TRIANGLE,None)


# cannyImg = cv2.Canny(thr,50,150,None,3,True)
#
# cornerHarris = cv2.cornerHarris(np.float32(img_gray),7,5,0.229)
# # thr = cv2.dilate(thr,kernel,iterations=2)

cv2.imshow("thr",thr)

img2,contuors,hierarchy = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)

cv2.namedWindow("w1",cv2.WINDOW_NORMAL)
cv2.resizeWindow("w1", w/2,h/2)

i=0
for cnt in contuors:
    approx = cv2.approxPolyDP(cnt,0.00005*cv2.arcLength(cnt,True),True)
    print  approx

    imgWithContuors = cv2.drawContours(img, [approx], -1, (255*np.random.random(),255*np.random.random(), 255*np.random.random()), -1)
    # imgWithContuors = cv2.polylines(img, approx,True,(i * np.random.random(), i * np.random.random(), i * np.random.random()),1,cv2.LINE_4)
    # cv2.putText(imgWithContuors, str(i),(approx[0][0][0],approx[0][0][1]),cv2.FONT_HERSHEY_COMPLEX,0.2,(0,0,255),1)
    i+=1

cv2.imshow("w1",imgWithContuors)
cv2.imwrite("../img_contuors.jpg",imgWithContuors)

cv2.waitKey(0)
cv2.destroyAllWindows()

exit()