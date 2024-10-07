import cv2

capture = cv2.VideoCapture(0)
capture.set(3,320)# 320 320 640 720
capture.set(4,240)#180 240  360 405

face_cascade = cv2.CascadeClassifier('C:\\anaconda3\\envs\\Emotion\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

while True:
        ret, img = capture.read()
        img = cv2.flip(img,1)#鏡表示にするため．
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x,y,w,h) in faces:
                cv2.circle(img,(int(x+w/2),int(y+h/2)),int(w/2),(0, 0, 255),2) # red

        cv2.imshow('img',img)
        # key Operation
        key = cv2.waitKey(10) 
        if key ==27 or key ==ord('q'): #escまたはqキーで終了
                break
capture.release()
cv2.destroyAllWindows()
print("Exit")      