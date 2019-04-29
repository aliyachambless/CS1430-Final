import cv2
import sys
from tensorflow import keras


video_capture = cv2.VideoCapture(0)

model = keras.models.load_model(model.h5)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_info = model.predict(frame)


    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30),
    #     flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    # )

    # # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', face_info)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
