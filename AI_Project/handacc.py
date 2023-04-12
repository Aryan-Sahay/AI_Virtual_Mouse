import cv2
import mediapipe as mp
import time

cp = cv2.VideoCapture(0)

hand_s = mp.solutions.hands
hands = hand_s.Hands()
mdraw = mp.solutions.drawing_utils

p_time = 0
c_time = 0

while True:
    success,img = cp.read()
    
    imgRgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRgb)

    if results.multi_hand_landmarks:
        for handl in results.multi_hand_landmarks:
            for i_d,l_m in enumerate(handl.landmark):
                h,w,p = img.shape
                py,px = int(l_m.x * w), int(l_m.y * h)
                print(i_d,px,py)
                
            mdraw.draw_landmarks(img,handl,hand_s.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1 /  (c_time - p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10,60), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (250,150,200),3)

    cv2.imshow("image",img)
    cv2.waitKey(1)
    