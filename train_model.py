import os
import cv2
import numpy as np
from PIL import Image

names=[]
paths=[]

for users in os.listdir("dataset"):
    names.append(users)

for name in names:
    for image in os.listdir("dataset/{}".format(name)):
        path_string = os.path.join("dataset/{}/".format(name), image)
        #dataset/chris/1_1.jpg
        paths.append(path_string)

#print(paths)
#till here we get images and path of images , after this we have to convert imgs into num bit array

faces = []
ids = []


for img_path in paths:  #here we are taking image path from path list
    image = Image.open(img_path).convert("L")
    #then we open that image uisg the open function and convert it into black and white using convert method
    imgNp = np.array(image,"uint8")

    faces.append(imgNp) 

    id = img_path.split("/")[2].split("_")[0]

    #print(id)
    
    ids.append(id)
    
ids = np.array(ids,dtype=np.int32)
labels = cv2.UMat(ids)

trainer = cv2.face.LBPHFaceRecognizer_create()
trainer.train(np.array(faces, dtype=object), ids)

trainer.write("training.yml")