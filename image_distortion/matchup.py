import numpy as np
from scipy.spatial import distance
import cv2
import math

"""
Matches up a filter_image and an image of a face. The filter_image
should have its nose directly in the center, and should be a square.
0 values in filter_image for spots where the original image should
show through it.

@param face_image np array face image
@param face_features_dictionary np array dictionary with point entries for
face features 'left_eye', 'right_eye', 'nose', and 'bottom'
@param filter_image np array filter image
@param filter_features_dictionary np array dictionary with point entries for
filter features 'left_eye', 'right_eye', 'nose', and 'bottom'
@return np array of original face image with filter superimposed on it
"""
def matchup(face_image, face_features_dictionary, filter_image, filter_features_dictionary):
    face_left_eye, face_right_eye, face_nose, face_bottom = face_features_dictionary['left_eye'], face_features_dictionary['right_eye'], face_features_dictionary['nose'], face_features_dictionary['bottom']
    filter_left_eye, filter_right_eye, filter_nose, filter_bottom = filter_features_dictionary['left_eye'], filter_features_dictionary['right_eye'], filter_features_dictionary['nose'], filter_features_dictionary['bottom']

    # point directly between eyes
    face_eye_center = np.array([(face_left_eye[0] + face_right_eye[0]) / 2, (face_left_eye[1] + face_right_eye[1]) / 2])
    filter_eye_center = np.array([(filter_left_eye[0] + filter_right_eye[0]) / 2, (filter_left_eye[1] + filter_right_eye[1]) / 2])
    # vector pointing from nose to center
    face_down_to_up = face_eye_center - face_nose
    filter_down_to_up = filter_eye_center - filter_nose
    # angle between orientation and straight vertical
    face_orientation = angle_between(face_down_to_up, np.array([1,0]))
    filter_orientation = angle_between(filter_down_to_up, np.array([1, 0]))
    # distance between eyes
    face_horiz_scale = distance.euclidean(face_left_eye, face_right_eye)
    filter_horiz_scale = distance.euclidean(filter_left_eye, filter_right_eye)
    # distance between eye center and bottom
    face_vert_scale = distance.euclidean(face_eye_center, face_bottom)
    fitler_vert_scale = distance.euclidean(filter_eye_center, filter_bottom)

    # creates a rescaled version of the filter image
    rescale = np.array(cv2.resize(np.float32(filter_image), 
        None, 
        fx=(face_horiz_scale / filter_horiz_scale), 
        fy=(face_vert_scale / fitler_vert_scale)))

    # calculates difference between filter orientation and face orientation
    rotation_diff = filter_orientation - face_orientation

    # rotates rescaled filter to orient with face
    num_rows, num_cols = rescale.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), math.degrees(rotation_diff), 1)
    rotated_filter = np.array(cv2.warpAffine(rescale, rotation_matrix, (num_cols, num_rows)))
    rotated_filter_height, rotated_filter_width = rotated_filter.shape[:2]

    # imposes filter on face and returns new mashup image
    rotated_center = (np.array(rotated_filter.shape) / 2).astype(int)
    filter_over_spots = np.array(np.where(rotated_filter != 0))
    # print(rotated_filter.shape)
    # print(rotated_filter.shape)
    # print(face_image.shape)
    for index in range(len(filter_over_spots[0])):
        point = [filter_over_spots[0][index], filter_over_spots[1][index]]
        displacement_from_center = [rotated_center[0] - point[0], rotated_center[1] - point[1]]
        try:
            face_image[face_nose[0] - displacement_from_center[0]][face_nose[1] - displacement_from_center[1]] = rotated_filter[point[0], point[1]]
            # print('success')
        except IndexError:
            # this will sometimes happen from resizing the filter so it's
            # too large; we want to just ignore these cases
            print("INDEX ERROR HIT")
            0

    return face_image

"""
Overlays a pixel value on a base pixel; if overlay value is 0, simply leaves the original
value there.

@param overlay the overlay pixel
@param base the base pixel
@return the result of overlaying the pixel
"""
def overlay_pixel_value(overlay, base):
    if (overlay == 0):
        return base
    else:
        return overlay

"""
implementation from
https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
"""
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

"""
0 (same) <= angle <= pi (negated)
"""
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))