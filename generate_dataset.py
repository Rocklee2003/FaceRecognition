import cv2
from pathlib import Path

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#img = cv2.imread("images/portrait1.jpg")

vc = cv2.VideoCapture(0)
#This line creates a VideoCapture object using cv2.VideoCapture(). The argument 0 specifies the default webcam as the video source.
#  If you have multiple cameras connected to your system, you can use different numbers (e.g., 1, 2, etc.) to select the desired camera.

print("Enter the ID and name of the person :")
userId = input()
userName = input()

count = 1

def saveImage(img, userName, userId,imgId):
    Path("dataset/{}".format(userName)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite("dataset/{}/{}_{}.jpg".format(userName, userId, imgId), img)


while True:
    _, img = vc.read()
    #Inside the while loop, this line reads a frame from the video source using the read() method of the VideoCapture object. The read() method returns two values: a boolean indicating whether the frame was read successfully and the frame itself.
    #In this case, we're only interested in the frame, so we discard the boolean value by using the underscore (_) as the variable name.
    
    originalImg=img
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray_img,scaleFactor=1.2,minNeighbors=5,minSize=(50,50))

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x, y),(x+w, y+h), (0, 255, 0), 2)
        coords = [x,y,w,h]

    cv2.imshow("identified Faces", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if count <= 20:
            roi_img= originalImg[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2] ]
            saveImage(roi_img,userName,userId,count)
            count += 1
        else:
            break
    elif key == ord('q'):
        break

vc.release()
#After the while loop ends, this line releases the video capture resource using the release() method of the VideoCapture object. 
# This is important to free up the camera resource when the program is finished.
cv2.destroyAllWindows()