import os
import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 200)

imgBackground = cv2.imread('Resources/Background.png')

# importing all mode images to a list
folderPathModes = "Resources/Modes"
listImgModesPath = os.listdir(folderPathModes)
listImgModes = []
for imgModePath in listImgModesPath:
    listImgModes.append(cv2.imread(os.path.join(folderPathModes, imgModePath)))
# print(listImgModes)

# importing all the icons to a list
folderPathIcons = "Resources/Icons"
listImgIconsPath = os.listdir(folderPathIcons)
listImgIcons = []
for imgIconPath in sorted(listImgIconsPath):
    listImgIcons.append(cv2.imread(os.path.join(folderPathIcons, imgIconPath)))

modeType = 3
selection =-1
counter =0
selection_speed=10
detector = HandDetector(detectionCon=0.8, maxHands=1)
modePositions=[(1565,276), (1390, 520), (1565,790)]
counterPause= 0
selectionList=[-1, -1, -1]



while True:
    success, img = cap.read()
    img = cv2.resize(img, (846, 637))
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)  # with draw
    imgBackground[185:185+637, 72:72+846] = img
    resized_mode = cv2.resize(listImgModes[modeType], (456,962))
    imgBackground[0:962, 1250:1710] = resized_mode

    # Find the hand and its landmarks

    if hands and counterPause == 0 and modeType!=0:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right
        fingers1 = detector.fingersUp(hand1)
        # print(fingers1)

        if fingers1 == [0,1,0,0,0]:
            if selection!=1:
                counter =1
            selection=1

        elif fingers1 == [0,1,1,0,0]:
            if selection!=2:
                counter =1
            selection=2
        elif fingers1 == [0,1,1,1,0]:
            if selection!=3:
                counter =1
            selection=3
        else:
            selection=-1
            counter=0

        if counter>0:
            counter+=1
            # print(counter)
            cv2.ellipse(imgBackground, modePositions[selection-1], (130,130), 0, 0, counter*selection_speed, (0,0,0), 20)
            if counter*selection_speed>360:
                modeType-=1
                selectionList[modeType] = selection
                counter =0
                selection=-1
                counterPause = 1


# to pause for a while after a selection is made
    if counterPause > 0:
        counterPause += 1
        if counterPause > 40:
            counterPause = 0


    if selectionList[2] != -1:
        listImgIcons[selectionList[2]-1] = cv2.resize(listImgIcons[selectionList[2]-1], (66, 66))
        imgBackground[855:855+66, 194:194+66]= listImgIcons[selectionList[2]-1]

    if selectionList[1] != -1:
        listImgIcons[selectionList[1]+2] = cv2.resize(listImgIcons[2+selectionList[1]], (66, 66))
        imgBackground[855:855+66, 466:466+66]= listImgIcons[selectionList[1]+2]

    if selectionList[0] != -1:
        listImgIcons[selectionList[0]+5] = cv2.resize(listImgIcons[selectionList[0]+5], (66, 66))
        imgBackground[855:855+66, 736:736+66]= listImgIcons[selectionList[0]+5]


    # Displaying image
    cv2.imshow("Background", imgBackground)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
