import cv2
import time
from numpy.lib.function_base import interp
import HandModule as hm
import numpy as np
import autopy

h_cam , w_cam = 270 , 330
frames = 100    
smooth = 10

p_time = 0
plocx,plocy = 0, 0
clocx,clocy = 0, 0

cp = cv2.VideoCapture(0)
cp.set(3,w_cam)
cp.set(4,h_cam)
detector = hm.hand_detector(maxhands = 1)
w_sc , h_sc = autopy.screen.size()


while True:

    #finding hand position
    sucess , img = cp.read()
    img = detector.find_hands(img)
    lm_list ,b_box = detector.find_position(img)
    
    #locating tip of index and middle fingure
    if (len(lm_list) != 0):
        x1,y1 = lm_list[8][1:]
        x2,y2 = lm_list[12][1:]

        #if fingures are visible
        finger = detector.fingersUp()
        cv2.rectangle(img, (frames,frames), (w_cam -frames, h_cam - frames), (250,160,160), 2)

        #moving index finger
        if (finger[1] == 1 and finger[2] == 0):

            #converting co-ordinates
            x3 = np.interp(x1, (frames, w_cam - frames), (0, w_sc))
            y3 = np.interp(y1, (frames, h_cam - frames), (0, h_sc))

            #smoothing the values
            clocx = plocx + (x3 - plocx) / smooth
            clocy = plocy + (y3 - plocy) / smooth

            #moving the curser
            autopy.mouse.move(w_sc - clocx, clocy)
            cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.cv2.FILLED)
            plocx , plocy = clocx , clocy

        #using fingers for clicking action
        if (finger[1] == 1 and finger[2] == 1):
            
            #distance between fingers
            length, img, l_info = detector.find_distance(8, 12, img)
            #print(length)

            #clicking action
            if (length <= 15):
                cv2.cv2.circle(img, (l_info[4],l_info[5]), 10, (255,160,160), cv2.cv2.FILLED)
                autopy.mouse.click()


    #fps
    c_time = time.time()
    fps = 1 /  (c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (15,60), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (180,180,255), 3)

    #final Display
    cv2.imshow("image",img)
    cv2.waitKey(1)
    
