import cv2 
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from image_distortion.matchup import matchup as MU
import argparse
import pickle
# import Image

# get the pipeline to transform this bby back to 96 by 96
with open('my_sklearn_objects.pkl', 'rb') as f:
    output_pipe = pickle.load(f)

# This function loads the correct filter image based on the user's command line argument
# run the code using python webcam_test_1.py -f <filter image file name>
# our given options are multi and plant
def load_data(file_name): 
    filter_img ="multi.png"
    if file_name == "multi":
        pass
    elif file_name == "plant":
        filter_img = "plant2.png"
    elif file_name == "std":
        filter_img = "mockup.png"
    return filter_img

# create the command line parser
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filter", required=True, help="Either multi.png, or plant2.png. Choose which face filter to use")
args = parser.parse_args()

# Load in the filter image based on command line input
filter_image = (cv2.imread(load_data(args.filter))).astype(int)


video_capture = cv2.VideoCapture(0)
# skip first frame of 0's
if_not_first = False
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = keras.models.load_model("finalmodel.h5")

filter_dict = {}
filter_y, filter_x = filter_image.shape[0], filter_image.shape[1]
filter_dict['left_eye'] = [(25 / 96) * filter_y, (27 / 96) * filter_x]
filter_dict['right_eye'] = [(25 / 96) * filter_y, (67 / 96) * filter_x]
filter_dict['nose'] = [(45 / 96) * filter_y, (47 / 96) * filter_x]
filter_dict['bottom'] = [(74 / 96) * filter_y, (47 / 96) * filter_x]


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    full_image = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # replicating input image to crop and feed into predictor 
    img_crop = gray 
    width, height, three = full_image.shape

    if (if_not_first) :
        # gets rectangle of face
        faces = face_cascade.detectMultiScale(img_crop, 1.3, 5)

        for x,y,w,h in faces :
            #crops rectangle out of input image
            img_crop = gray[y:y+h, x:x+w]

            #save scaling factor used to make it square
            scaleX = w / 96
            scaleY = h / 96
            img_crop_fullsize = np.copy(frame[y:y+h, x:x+w, :])
            img_crop_half_length = int(img_crop_fullsize.shape[0] / 2.0)
            try:
                img_crop = cv2.resize(img_crop, dsize=(96, 96), interpolation = cv2.INTER_CUBIC)
            except:
                # print("BIGG")
                continue
            X = np.reshape(img_crop, (96, 96))
            X = X.astype(np.float32)
            X = X / 255.

            #predicting with model
            predictions = model.predict(X[np.newaxis, :, :, np.newaxis])
            face_info = (output_pipe.inverse_transform(predictions).reshape(1, 30)-48)/48


            # plot puts a filter over the face in the webcam feed
            def plot (fig):

                # sets up the face dictionary based on
                fig = fig * img_crop_half_length + img_crop_half_length
                face_dict = {}
                face_dict['right_eye'] = [int(fig[0][1]), int(fig[0][0])]
                face_dict['left_eye'] = [int(fig[0][3]), int(fig[0][2])]
                face_dict['nose'] = [int(fig[0][21]), int(fig[0][20])]
                face_dict['bottom'] = [int(fig[0][29]), int(fig[0, 28])]

                img_crop_fullsize[:, :, 0] = MU(img_crop_fullsize[:, :, 0], face_dict, filter_image[:, :, 0], filter_dict)
                img_crop_fullsize[:, :, 1] = MU(img_crop_fullsize[:, :, 1], face_dict, filter_image[:, :, 1], filter_dict)
                img_crop_fullsize[:, :, 2] = MU(img_crop_fullsize[:, :, 2], face_dict, filter_image[:, :, 2], filter_dict)
                i = 0
                # rescaling to cropped rectangle shape, then shifting by x and y to map to right place on full_image
                while i < 29:
                    fig[0][i] = int((fig[0][i] * scaleX) + x)
                    fig[0][i + 1] = int((fig[0][i + 1] * scaleY) + y)
                    i = i + 2

                return img_crop_fullsize

            # place the applied-filter face where the old face sat
            fixed_face = plot(face_info)
            full_image[y:y+h, x:x+w, 0] = fixed_face[:, :, 0]
            full_image[y:y+h, x:x+w, 1] = fixed_face[:, :, 1]
            full_image[y:y+h, x:x+w, 2] = fixed_face[:, :, 2]

        # show the image
        cv2.imshow('Video', full_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if_not_first = True

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
