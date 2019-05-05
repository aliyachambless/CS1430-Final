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
if_not_first = False
face_cascade = cv2.CascadeClassifier('/Users/victoriaxu/Documents/CS/opencv-4.0.0/data/haarcascades/haarcascade_frontalface_default.xml')


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ###############################################################################################
    img_crop = gray 
    ###############################################################################################
    width, height = gray.shape
    print("DIMENSIONS" + str(width) + str(height))
    if (if_not_first) :
    ###############################################################################################
        faces = face_cascade.detectMultiScale(img_crop, 1.3, 5)
        print("LENGTH OF FACES + " + str(len(faces)))
        for x,y,w,h in faces :
            print("THIS IS X", str(x))
            print("THIS IS W", str(w))
            print("THIS IS width", str(width))
            print("THIS IS height", str(height))    
            # if (x + w > height) | (y + h > width) :
            #     x = x - (x + w - height)
            #     y = y - (y + h - width)
            #     ("PRINT THE IF HIT !!! ")
            print("THIS IS X", str(x))
            print("THIS IS W", str(w))
            print("THIS IS Y", str(y))
            print("THIS IS H", str(h))
            # side = max(w, h)
            # if (side < 96):
            #     side = 96
            print("GRAY IN FOR : " + str(img_crop))
            img_crop = img_crop[y:y+h, x:x+w]
            print("help x "+ str(x))
            print("help x + w "+ str(x+w))
            print("help y "+ str(y))
            print("help y+h "+ str(y+h))

            print("GRAY AFTER CROPPING????? " + str(img_crop.shape))
            cv2.rectangle(gray,( x, y),( x + w,  y + h),(255,0,0),2)
            # if (gray.shape[0] == 0) | (gray.shape[1] == 0) :
            #     gray = img[0:96, 0:96]
            #     print("BAD CASE WAS HIT!!!! ")
            print("THIS IS THE SHAPE BEFORE RESIZE   " + str(gray.shape))
            ###############################################################################################
            scaleX = w / 96
            scaleY = h / 96
            img_crop = cv2.resize(img_crop, dsize=(96, 96), interpolation = cv2.INTER_CUBIC)
            ###############################################################################################
            X = np.reshape(img_crop, (1, 9216))
            X = X.astype(np.float32)
            X = X / 255.

            model = keras.models.load_model("model2.h5")

            face_info = model.predict(X)
###############################################################################################
            def plot (fig) :
                fig = fig * 48 
                #math it in relation to the original image: add x and y? 
                i = 0
                while i < 29 :
                    cv2.circle(gray, (int((fig[0][i + 1] + y) * scaleY) , int((fig[0][i] + x) + scaleX)), 3, (255,0,0))
                    # print("HIT")
                    i = i + 1
                return gray
###############################################################################################

            # Display the resulting frame
            # im = fig2img ( figure )
            # im.show()
            cv2.imshow('Video', plot(face_info))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if_not_first = True

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
