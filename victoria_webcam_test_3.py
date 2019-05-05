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
from scipy.spatial import distance as dt 
# import Image



video_capture = cv2.VideoCapture(0)
if_not_first = False
face_cascade = cv2.CascadeClassifier('/Users/ambikamiglani/Downloads/opencv-4.0.0/data/haarcascades/haarcascade_frontalface_default.xml')


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ###############################################################################################
    img_crop = gray 
    ###############################################################################################
    width, height = gray.shape
  
    if (if_not_first) :
    ###############################################################################################
        faces = face_cascade.detectMultiScale(img_crop, 1.3, 5)
        faces= list(faces)
        next= [200, 300, 200, 200]
        faces.append(next)
        faces= tuple(faces)
        # print("faces" + str(faces))
        # print("faces shape" + str(len(faces)))

        print("LEN FACES BEFORE" + str(len(faces)))

        if(len(faces) > 1): 
            faces= faces[0]
            faces= [tuple(faces)]
            print("faces" + str(tuple(faces)))
            # np.asarray(faces)

        print("LEN FACES" + str(len(faces)))

        for x,y,w,h in faces :

            # cv2.rectangle(gray,( x, y),( x + w,  y + h),(255,0,0),4)

            # if(len(faces)> 1): 
           
            #     closest_face= faces[0]
            #     x= closest_face[0]
            #     y= closest_face[1]
            #     w= closest_face[2]
            #     h= closest_face[3]

                     # dist1= dt.euclidean(prev_face.flatten, np.flatten(faces[0]))
                # dist2= dt.euclidean(np.flatten(prev_face), np.flatten(faces[1]))
                # dist3= dt.euclidean(np.flatten(prev_face), np.flatten(faces[2]))
                # closest_face= min(dist1, dist2, dist3) 
                # cv2.rectangle(gray,( cx, cy),( cx + cw,  cy + ch),(220,20,60),6)
                # img_crop_c = img_crop_c[cy:cy+ch, cx:cx+cw]/

                # print("CLOSEST" + str(closest_face))
                
            cv2.rectangle(gray,( x, y),( x + w,  y + h),(255,0,0),4)
            img_crop = img_crop[y:y+h, x:x+w]
            print("X2" + str(x))
            print("Y2" + str(y))
            print("W2" + str(w))
            print("H2" + str(h))
            # print("dis is imgcrop" + str(img_crop))
            # cv2.rectangle(gray,( x, y),( x + w,  y + h),(255,0,0),4)
            # if (gray.shape[0] == 0) | (gray.shape[1] == 0) :
            #     gray = img[0:96, 0:96]
            #     print("BAD CASE WAS HIT!!!! ")
            
            ###############################################################################################
            img_crop = cv2.resize(img_crop, dsize=(96, 96), interpolation = cv2.INTER_CUBIC)
            scaleX = w / 96
            scaleY = h / 96
            ###############################################################################################
            X = np.reshape(img_crop, (1, 9216))
            X = X.astype(np.float32)
            X = X / 255

            model = keras.models.load_model("model2.h5")

            face_info = model.predict(X)
###############################################################################################
            def plot (fig) :
                fig = fig * 48 
                #math it in relation to the original image: add x and y? 
                i = 0
                while i < 29 :
                    cv2.circle(gray, (int((fig[0][i + 1]* scaleX) + x) , int((fig[0][i] * scaleY) + y)), 3, (255,0,0))
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
