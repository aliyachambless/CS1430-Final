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
    # img = frame.apply(lambda im: np.fromstring(im, sep=' '))
    # if cols:  # get a subset of columns
    #     frame = frame[list(cols) + img]
    # print(df.count())  # prints the number of values for each column
    # frame = frame.dropna()  # drop all rows that have missing values in them
    # X = np.vstack(img.values) / 255.  # scale pixel values to [0, 1]
    # X = X.astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    width, height = gray.shape
    print("DIMENSIONS" + str(width) + str(height))
    if (if_not_first) :
        # print(gray)
        
    #VVV: we probably should do the viola jones cascade face crop (the box around face)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # print("THIS IS THE FACES" + str(faces[0]))
        for x,y,w,h in faces :
            # print("THIS IS X"  str(x))
            # print("THIS IS Y" + str(y))
            # print("WIDTH" + str(w))
            # print("HEIGHT" + str(h))

        # if (faces[0].size != 0) : 
        #     print("THIS IS THE FACES" + str(faces))
            # x = faces[0]
            # y = faces[1]
            # w = faces
            # h = faces[3]
            # x, y, w, h = faces
            # print(x)
            # print(y)
            # print(w)
            # print(h)
            # img = cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            
            # side = max(w, h)
            # print("this is the side length" + str(side))     

            # img_crop = gray[y:y+h, x:x+w]
            prev_face= cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),4)
            # print("Prev fac" + str( prev_face))
            # print("prev face SHAPE"  + str(prev_face.shape))
            
            # print("prev fce" + str(np.ndarray.astype(prev_face)))
            # gray= gray[y:y+h, x:x+w]
            print("FACES LENGTH" + str(len(faces)))
            # print("GRAY SAHAPE" + str(gray.shape))
            # print("GRAY IN FOR : " + str(gray))
            # if(gray.shape[0])

            if(len(faces)> 1): 
                dist1= dt.euclidean(prev_face, faces[0]) 
                dist2= dt.euclidean(prev_face, faces[1])
                dist3= dt.euclidean(prev_face, faces[2])
                closest_face= min(dist1, dist2, dist3) 
                print("CLOSEST" + str(closest_face))
            
            
            # scaleX = w / 96
            # scaleY = h / 96
            # img_crop = cv2.resize(img_crop, dsize=(96, 96), interpolation = cv2.INTER_CUBIC)

            gray = cv2.resize(gray, dsize=(96, 96), interpolation = cv2.INTER_CUBIC)
            X = np.reshape(gray, (1, 9216))
            X = X.astype(np.float32)
            X = X / 255

            model = keras.models.load_model("model2.h5")
            face_info = model.predict(X)
            

        def plot (fig) :
            fig = fig * 48 + 48
            i = 0
            while i < 29 :
                cv2.circle(gray, (fig[0][i], fig[0][i + 1]), 3, (255,0,0))
                # print("HIT")
                i = i + 1
            return gray


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
