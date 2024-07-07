import cv2
import os


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training.yml")

names=[]
for users in os.listdir("dataset"):
    names.append(users)

img = cv2.imread("test/chris.jpg")


while True:
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray_img,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(50,50))

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x, y),(x+w, y+h), (0, 255, 0), 2)
        id,_ = recognizer.predict(gray_img[y:y+h, x:x+w])
        if id:
            cv2.putText(img, names[id-1],(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
        else:
            cv2.putText(img, "unknown",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)

    cv2.imshow("Recognize Faces", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()