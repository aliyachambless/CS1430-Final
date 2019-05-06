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
from image_distortion.matchup import matchup as MU
# import Image



video_capture = cv2.VideoCapture(0)
#skip first frame of 0's
if_not_first = False
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = keras.models.load_model("model2.h5")
filter_image = cv2.cvtColor(cv2.imread('mockup.png'), cv2.COLOR_BGR2GRAY)
filter_dict = {}
filter_dict['left_eye'] = [25,27]
filter_dict['right_eye'] = [25,67]
filter_dict['nose'] = [45,47]
filter_dict['bottom'] = [74,47]


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
            # cv2.imwrite('/Users/victoriaxu/Downloads/face.png', img_crop)

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
                fig = fig * 48 + 48

                face_dict = {}
                face_dict['right_eye'] = [int(fig[0][1]), int(fig[0][0])]
                # cv2.circle(img_crop, (fig[0][0],fig[0][1]), 3, (255,0,0))
                # print(str(fig[0][0]) + " " + str(fig[0][1]))
                face_dict['left_eye'] = [int(fig[0][3]), int(fig[0][2])]
                # cv2.circle(img_crop, (fig[0][2], fig[0][3]), 3, (255,0,0))
                # print(str(fig[0][2]) + " " + str(fig[0][3]))

                face_dict['nose'] = [int(fig[0][21]), int(fig[0][20])]
                # cv2.circle(img_crop, (fig[0][20], fig[0][21]), 3, (255,0,0))
                # print(str(fig[0][0]) + " " + str(fig[0][1]))

                face_dict['bottom'] = [int(fig[0][29]), int(fig[0,28])]
                # cv2.circle(img_crop, (fig[0][28],fig[0,29]), 3, (255,0,0))
                # print(str(fig[0][0]) + " " + str(fig[0][1]))

                # cv2.circle(gray, (int((fig[0][i] * scaleX) + x), int((fig[0][i + 1] * scaleY) + y)), 3, (255,0,0))
                # img_crop[:][:][0] = MU(img_crop[:][:][0], face_dict, filter_image, filter_dict)
                # print(img_crop[:][:][0].shape)
                # img_crop[:][:][1] = MU(img_crop[:][:][1], face_dict, filter_image, filter_dict)
                # img_crop[:][:][2] = MU(img_crop[:][:][2], face_dict, filter_image, filter_dict)
                # final = img_crop
                # print("BEFORE " + str(img_crop))
                final = MU(img_crop, face_dict, filter_image, filter_dict)
                print("EQUALITY " + str(img_crop == final))


                # final = np.array([MU(img_crop[:][0][:], face_dict, filter_image, filter_dict), MU(img_crop[:][1][:], face_dict, filter_image, filter_dict), MU(img_crop[:][2][:], face_dict, filter_image, filter_dict)])
                print("BEGINNNIG ")
                print(img_crop.shape)
                print(face_dict)
                print(filter_image.shape)
                print(filter_dict)
                print(final.shape)
                print(gray.shape)
                #rescaling face_info coords to 96x96 square
                i = 0
                #rescaling to cropped rectangle shape, then shifting by x and y to map to right place on gray
                while i < 29 :
                    # cv2.circle(gray, (int((fig[0][i] * scaleX) + x), int((fig[0][i + 1] * scaleY) + y)), 3, (255,0,0))
                    # print("DIMENSIONS "+ str(gray.shape))
                    # print("WHAT IS "+ str((int((fig[0][i] * scaleX) + x), int((fig[0][i + 1] * scaleY) + y))))
                    fig[0][i] = int((fig[0][i] * scaleX) + x)
                    fig[0][i + 1] = int((fig[0][i + 1] * scaleY) + y)
                    # print("HIT")
                    i = i + 2


                return final

            # Display the resulting frame
            gray[y:y+h, x:x+w] = cv2.resize(plot(face_info), (w,h))
            cv2.imshow('Video', gray)

            # cv2.imshow('Video', plot(face_info))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if_not_first = True

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
