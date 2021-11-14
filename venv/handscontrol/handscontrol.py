import cv2
import numpy as np


captura = cv2.VideoCapture(0)

# lb = np.array ([15,100,100])
# ub = np.array ([100,255,255])
lb= np.array([0, 0, 0])
ub = np.array([350,55,100])
kernelOpen = np.ones((0,0))
kernelClose = np.ones((50,50))

while (1):
    ret, frame = captura.read()
    flipped = cv2.flip(frame,1)
    flipped = cv2.resize(flipped,(500,400))
    imgSeg = cv2.cvtColor(flipped,cv2.COLOR_BGR2HSV)
    cv2.imshow("gray",imgSeg)
    mask = cv2.inRange(imgSeg,lb,ub)
    # maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    # maskClose = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelClose)
    cv2.imshow("mask",mask)
    contours,he= cv2.findContours(mask,cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)
    if(len(contours)!=0):
        # print(contours)
        b = max(contours, key=cv2.contourArea)
        west = tuple(b[b[:, :, 0].argmin()][0])
        east = tuple(b[b[:, :, 0].argmax()][0])
        north = tuple(b[b[:, :, 1].argmin()][0])
        south= tuple(b[b[:, :, 1].argmax()][0])
        center_x = (west[0]+east[0])/2
        center_y= (north[0]+south[0])/2

        cv2.drawContours(flipped, b,-1,(0,255,0),1)
        cv2.circle(flipped,west,6,(0,0,255), -1)
        cv2.circle(flipped,east,6,(0,0,255),-1)
        cv2.circle(flipped, north, 6, (0, 0, 255), -1)
        cv2.circle(flipped, south, 6, (0, 0, 255), -1)
        cv2.circle(flipped, (int(center_x),int(center_y)), 6, (255, 0, 0), -1)


    cv2.imshow("Video", flipped)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

captura.release()
cv2.destroyAllWindows()