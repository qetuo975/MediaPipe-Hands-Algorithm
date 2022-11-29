import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
    
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
            

    # sağ el 250
    # sol el 450
    
    
    if len(lmList) != 0:
        fingers = []
        
        """
        if lmList[tipIds[0]][1] > lmList[tipIds[1]][1]:
            print("Sağ El")
            
        if lmList[tipIds[0]][1] < lmList[tipIds[1]][1]:
            print("Sol El")
            
        print("--------------------------")
        print("Baş Parmak ",lmList[tipIds[0]])
        print("İşaret Parmak ",lmList[tipIds[1]])
        print("Orta Parmak ",lmList[tipIds[2]])
        print("Nişan Parmak ",lmList[tipIds[3]])
        print("Serçe Parmak ",lmList[tipIds[4]])
        print("--------------------------")
        """  
        
        # Sağ ve Sol El Kordinatlarına Bakarak Koşulları Daha Kesinleştirmeye Çalış
        
        if lmList[tipIds[0]][1] < lmList[tipIds[1]][1]:
            print("Sol El")
            # BasParmak
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            
            # 4 Parmak
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
            totalF = fingers.count(1)
            cv2.putText(img, str(totalF), (30,125), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 8)
            
        if lmList[tipIds[0]][1] > lmList[tipIds[1]][1]:
            print("Sağ El")
                
            # BasParmak
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            
            # 4 Parmak
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
            totalF = fingers.count(1)
            cv2.putText(img, str(totalF), (30,125), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 8)
            
        if fingers.count(1) == 0:
            cv2.putText(img, str("Saldırganlık Algılandı."), (60,250), cv2.FONT_HERSHEY_PLAIN, 10, (0,255,0), 8)
            
            
        print("--------------------------")
        print("Baş Parmak ",lmList[tipIds[0]])
        print("İşaret Parmak ",lmList[tipIds[1]])
        print("Orta Parmak ",lmList[tipIds[2]])
        print("Nişan Parmak ",lmList[tipIds[3]])
        print("Serçe Parmak ",lmList[tipIds[4]])
        print("--------------------------")
        
        
    cv2.imshow("AI", img)
    cv2.waitKey(1)
