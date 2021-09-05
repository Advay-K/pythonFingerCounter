import cv2
import os
import time
import handTrack as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

fpath = "f_images"
myList = os.listdir(fpath)
#print (myList)
overlayList = []

for imgPath in myList:
    image = cv2.imread(f'{fpath}/{imgPath}')
    image = cv2.resize(image, (200, 200))
    overlayList.append(image)

#print (len(overlayList))
pTime = 0

detector = htm.handDetector(maxHands=1)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, frame = cap.read()
    detector.findHands(frame)
    lmList = detector.findPosition(frame, draw = False)

    if len(lmList) != 0:
        fingers = []

        #thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #4 fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print (fingers)
        totalFingers = fingers.count(1)



        h, w, c = overlayList[totalFingers-1].shape
        frame[0:h, 0:w] = overlayList[totalFingers-1]

        cv2.rectangle(frame, (20, 225), (170, 425), (255, 0, 100), -1)
        cv2.putText(frame, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_TRIPLEX, 5, (255, 255, 255), 15)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (450, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 100), 2)

    cv2.imshow("image", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()