#THIS IS BRENDANS NOTES
#ITS FAILING WHEN IT DETECTS MORE THAN ONE FACE (BC THEN ITS TRYING TO ACCESS ALREADY SPLICED ARRAY POINTS)
#A SOLUTION: MAKE SURE TO HAVE AN ORIGINAL IMAGE SO THAT IT DOESN'T TOTALLY ALTER ORIGINAL
    #THIS WOULD ALSO ADDRESS THE ZOOM IN/ZOOM OUT ISSUE
#FOR TESTING: LETS JUST USE RECTANGLE BEFORE LOCATING WHERE TO CROP


import cv2 
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
# import Image



video_capture = cv2.VideoCapture(0)
#skip first frame of 0's
if_not_first = False
face_cascade = cv2.CascadeClassifier('/Users/victoriaxu/Documents/CS/opencv-4.0.0/data/haarcascades/haarcascade_frontalface_default.xml')
model = keras.models.load_model("model2.h5")


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #replicating input image to crop and feed into predictor 
    img_crop = gray 
    width, height = gray.shape
    if (if_not_first) :
        #gets rectangle of face
        faces = face_cascade.detectMultiScale(img_crop, 1.3, 5)
        for x,y,w,h in faces :
            #crops rectangle out of input image
            img_crop = img_crop[y:y+h, x:x+w]
            cv2.rectangle(gray,( x, y),( x + w,  y + h),(255,0,0),2)
            #save scaling factor used to make it square
            scaleX = w / 96
            scaleY = h / 96
            img_crop = cv2.resize(img_crop, dsize=(96, 96), interpolation = cv2.INTER_CUBIC)
            X = np.reshape(img_crop, (1, 9216))
            X = X.astype(np.float32)
            X = X / 255.

            #predicting with model
            face_info = model.predict(X)

            #defining plot
            def plot (fig) :
                #rescaling face_info coords to 96x96 square
                fig = fig * 48 + 48
                i = 0
                #rescaling to cropped rectangle shape, then shifting by x and y to map to right place on gray
                while i < 29 :
                    cv2.circle(gray, (int((fig[0][i] * scaleX) + x), int((fig[0][i + 1] * scaleY) + y)), 3, (255,0,0))
                    # print("HIT")
                    i = i + 1
                return gray

            # Display the resulting frame
            cv2.imshow('Video', plot(face_info))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if_not_first = True

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
