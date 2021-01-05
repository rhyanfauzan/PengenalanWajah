import cv2, os, numpy as np
wajahDir = 'datawajah'
latihDir = 'latihwajah'

cam = cv2.VideoCapture(0)
cam.set(3, 1200) #ubah lebar cam
cam.set(4, 700) #ubah tinggi cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read(latihDir+'/training.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak diketahui','Ryan Fauzan','Clarissa Des']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame,1) #vertikal flip
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.2, 5,minSize=(round(minWidth),round(minHeight)),) #frame, scaleFactor, min Neighbors
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,255),2)
        id, confidence = faceRecognizer.predict(abuAbu[y:y+h,x:x+w]) #confidence = 0 artinya cocok sempurna
        if confidence <=80 :
            nameID = names[id]
            confidenceTxt = " {0}%".format(round(100-confidence))
        else:
            nameID = names[0]
            confidenceTxt = " {0}%".format(round(100-confidence))
        cv2.putText(frame,str(nameID),(x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(frame,str(confidenceTxt),(x+5,y+h-5),font,1,(0,255,255),1)

    cv2.putText(frame, 'Sistem Multimedia', (200, 450), font, 1, (0, 255, 255))  # ---write the text
    cv2.imshow('Recognisi Wajah',frame)
    #cv2.imshow('Webcam - Grey',abuAbu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
print("EXIT")
cam.release()
cv2.destroyAllWindows()