import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
# import Image



video_capture = cv2.VideoCapture(0)
if_not_first = False


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
    if (if_not_first) :
        print(gray)
        face_cascade = cv2.CascadeClassifier('/Users/victoriaxu/Documents/CS/opencv-4.0.0/data/haarcascades/haarcascade_frontalface_default.xml')
    #VVV: we probably should do the viola jones cascade face crop (the box around face)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # print("THIS IS THE FACES" + str(faces[0]))
        for x,y,w,h in faces :
            print("THIS IS X", str(x))
        # if (faces[0].size != 0) : 
        #     print("THIS IS THE FACES" + str(faces))
            # x = faces[0]
            # y = faces[1]
            # w = faces[2]
            # h = faces[3]
            # x, y, w, h = faces
            # print(x)
            # print(y)
            # print(w)
            # print(h)
            # img = cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            side = max(w, h)
            if (side < 96):
                side = 96
            print("THIS IS SIDE : " + str(side))
            print("THIS IS GRAY being cropped : " + str(gray.shape))

            gray = gray[x:x+side, y:y+side]
            print("GRAY IN FOR : " + str(gray))
        gray = cv2.resize(gray, dsize=(96, 96), interpolation = cv2.INTER_CUBIC)
        X = np.reshape(gray, (1, 9216))
        X = X.astype(np.float32)
        X = X / 255.

        model = keras.models.load_model("model.h5")

        # model = tf.keras.models.load_model('model.h5')
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
