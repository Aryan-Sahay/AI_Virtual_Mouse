import cv2
import mediapipe as mp
import time
import math
import numpy as np 

class hand_detector():
    def __init__(self, mode = False, maxhands = 2, detectionscon = 0.5, trackcon = 0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectioncon = detectionscon
        self.trackcon = trackcon

        

        self.hand_s = mp.solutions.hands
        self.hands = self.hand_s.Hands(self.mode, self.maxhands, self.detectioncon, self.trackcon )
        self.mdraw = mp.solutions.drawing_utils
        self.tip_id = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw = True):

        imgRgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)

        if self.results.multi_hand_landmarks:
            for handl in self.results.multi_hand_landmarks:
                if draw:
                    self.mdraw.draw_landmarks(img,handl,self.hand_s.HAND_CONNECTIONS)
        return img   

    def find_position(self, img, handn = 0, draw = True):

        self.lm_list = []
        x_list = []
        y_list = []
        b_box = []  
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handn]

            for i_d,l_m in enumerate(myhand.landmark):
                    h,w,p = img.shape
                    px,py = int(l_m.x * w), int(l_m.y * h)
                    self.lm_list.append([i_d,px,py])
                    x_list.append(px)
                    y_list.append(py)   
                    
                    if draw:
                        cv2.circle(img, (px,py), 5, (255,0,0), cv2.FILLED)
            x_min , x_max = min(x_list) , max(x_list)
            y_min , y_max = min(y_list) , max(y_list)
            b_box = x_min, y_min, x_max, y_max

            if draw:
                cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0,255,0), 2)

        return self.lm_list , b_box

    def fingersUp(self):
        fingers = []

        #for thumb
        if self.lm_list[self.tip_id[0]][1] > self.lm_list[self.tip_id[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #for index finger
        for i_d in range(1,5):
            if self.lm_list[self.tip_id[i_d]][2] < self.lm_list[self.tip_id[i_d] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def find_distance(self, p1, p2, img, draw = True, r = 10, t = 3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        px, py = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), t)
            cv2.circle(img, (x1, y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (px, py), r, (0,0,255), cv2.FILLED) 
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, px, py]


def main():
    cp = cv2.VideoCapture(0)
    detector = hand_detector()

    p_time = 0
    c_time = 0

    while True:
        success,img = cp.read()
        img = detector.find_hands(img)
        lm_list, b_box = detector.find_position(img)

        if (len(lm_list)!= 0):
            print(lm_list[8])

        c_time = time.time()
        fps = 1 /  (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10,60), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (250,150,200),3)

        cv2.imshow("image",img)
        cv2.waitKey(1)  


if  __name__ == "__main__":
        main()