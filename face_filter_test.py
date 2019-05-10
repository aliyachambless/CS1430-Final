import cv2
# face_dict = {'right_eye': [36.708717, 68.66651], 'left_eye': [33.846825, 29.472046], 'nose': [57.41571, 46.345577], 'bottom': [84.589005, 47.083366]}
face_dict = {'right_eye': [int(36.708717), int(68.66651)], 'left_eye': [int(33.846825), int(29.472046)], 'nose': [int(57.41571), int(46.345577)], 'bottom': [int(84.589005), int(47.083366)]}
filter_dict = {'left_eye': [25, 27], 'right_eye': [25, 67], 'nose': [45, 47], 'bottom': [74, 47]}
filter_image = cv2.cvtColor(cv2.imread('mockup.png'), cv2.COLOR_BGR2GRAY)
face_image = cv2.cvtColor(cv2.imread('face.png'), cv2.COLOR_BGR2GRAY)

from image_distortion.matchup import matchup as MU
ontop = MU(face_image, face_dict, filter_image, filter_dict)

from skimage import io as SIO
SIO.imshow(ontop)
SIO.show()
