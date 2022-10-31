###########################################################################################################
############################################## Virtual Mouse ##############################################
###########################################################################################################

import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm #put the code HandTrackingModule in the same folder of this code
import math
import numpy as np
import autopy

##############################
wCam, hCam = 1280, 720
frameR = 100 # frame reduction
smoothening = 5
#############################

mpPose = mp.solutions.mediapipe.python.solutions.pose # for use mediapipe
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0) # test with 0,1,2 depend where is located your webcam
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
cTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.handDetector(maxHands = 1)
wScr, hScr = autopy.screen.size()

while True:
    # Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # Get the tip of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        #print (x1, y1, x2, y2)
        # check which finger are up
        fingers = detector.fingersUp()  #NEW ###########
        print(fingers)
        # only index finger: moving mode
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam-frameR), (255,0, 0), 2)
        if fingers[1] == 1 and fingers[2]==0:
            # convert coordinates
            x3 = np.interp(x1, (frameR ,wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR,hCam-frameR), (0, hScr))
            # smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # move mouse
            autopy.mouse.move(wScr-clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY
            # both index and middle fingers are up: clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12,img)
            print(length)
            # click mouse if distance short
            if length <40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (255, 0, 0), cv2.FILLED)
                autopy.mouse.click()     


    #frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #cv2.putText(img, 'FPS: ' + str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)


    # Display
    cv2.imshow('Image', img)
    cv2.waitKey(1)

