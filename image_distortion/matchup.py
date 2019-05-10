import numpy as np
from scipy.spatial import distance
import cv2
import math
import operator
from skimage import io as SIO

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
    orig_shape = face_image.shape
    # orig_face = np.copy(face_image)
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
    # print(rotated_filter.shape)
    # print(rotated_filter.shape)
    # print(face_image.shape)

    ########## START FAST ############

    # mask = np.ma.array(rotated_filter, mask=(rotated_filter < .0000001), fill_value = -5000.)
    mask = np.ma.array(rotated_filter, mask=(rotated_filter == 0), fill_value = -5000.)

    fixed_filter = mask.filled()
    filter_half_ht, filter_half_wd = np.array(fixed_filter.shape) / 2
    # face_image = - face_image
    # print(fixed_filter.shape)
    # print(face_image.shape)
    # print(filter_half_ht)
    # print(filter_half_wd)

    # sample fits in the face_image and does its best to center the shape of the filter on the location of the nose
    sample = -face_image[max(face_nose[0] - math.floor(filter_half_ht), 0):min(face_nose[0] + math.ceil(filter_half_ht), face_image.shape[0]),max(face_nose[1] - math.floor(filter_half_wd), 0):min(face_nose[1] + math.ceil(filter_half_wd), face_image.shape[1])]
    sample_half_ht, sample_half_wd = np.array(sample.shape) / 2

    # filter_overlay is a center-centered crop of filter which does its best to fit the filter on the face sample
    filter_overlay = fixed_filter[max(rotated_center[0] - math.floor(sample_half_ht), 0):min(rotated_center[0] + math.ceil(sample_half_ht),fixed_filter.shape[0]), max(rotated_center[1] - math.floor(sample_half_wd), 0):min(rotated_center[1] + math.ceil(sample_half_wd), fixed_filter.shape[1])]

    # 
    # print(sample.shape == filter_overlay.shape)

    # this is the shape of both the sample and the filter overlay
    small_shapes = np.minimum(np.array(sample.shape), np.array(filter_overlay.shape))
    assert (small_shapes[0] == sample.shape[0] == filter_overlay.shape[0] and
            small_shapes[1] == sample.shape[1] == filter_overlay.shape[1])
    # sample = cropND(sample, small_shapes)
    # filter_overlay = cropND(filter_overlay, small_shapes)

    # sample = crop2DCenter(sample, small_shapes, (np.array(sample.shape) / 2).astype(int))
    # filter_overlay = crop2DCenter(filter_overlay, small_shapes, (np.array(filter_overlay.shape) / 2).astype(int))

    # print(rotated_center[0] - math.floor(sample_half_ht))
    # print(rotated_center[0] + math.ceil(sample_half_ht))
    # print(sample.shape)
    # print(sample_half_ht)
    # print(sample_half_wd)
    # print(filter_overlay.shape)
    # print(rotated_center)
    # print(fixed_filter.shape)
    # print(small_shapes)
    # print(face_image.shape)
    # a = (face_nose[0] - math.floor(small_shapes[0] / 2))
    # b = (face_nose[0] + math.ceil(small_shapes[0] / 2))
    # c = (face_nose[1] - math.floor(small_shapes[1] / 2))
    # d = (face_nose[1] + math.ceil(small_shapes[1] / 2))
    # print((face_image[a:b,c:d]).shape)
    # print((face_nose[0] - math.floor(small_shapes[0] / 2)))
    # print((face_nose[0] + math.ceil(small_shapes[0] / 2)))
    # print((face_nose[1] - math.floor(small_shapes[1] / 2)))
    # print((face_nose[1] + math.ceil(small_shapes[1] / 2)))
    # print(face_image[face_nose[0] - math.floor(small_shapes[0] / 2):face_nose[0] + math.ceil(small_shapes[0] / 2)][face_nose[1] - math.floor(small_shapes[1] / 2):face_nose[1] + math.ceil(small_shapes[1] / 2)].shape)
    # print((face_nose[0] - math.floor(small_shapes[0] / 2)) - (face_nose[0] + math.ceil(small_shapes[0] / 2)))
    # print((face_nose[1] - math.floor(small_shapes[1] / 2)) - (face_nose[1] + math.ceil(small_shapes[1] / 2)))
    # print(face_nose)
    cast = np.absolute(np.maximum(-sample, filter_overlay))
    slowcast = np.copy(-sample)
    for i in range(len(filter_overlay)):
        for j in range(len(i)):
            

    # print(np.any(cast < 0))
    # SIO.imshow(cast)
    # SIO.show()

    # from skimage import io as SIO
    # SIO.imshow(cast)
    # SIO.show()
    # print((face_nose[0] - math.floor(small_shapes[0] / 2)) -  (face_nose[0] + math.ceil(small_shapes[0] / 2)))
    # print((face_nose[1] - math.floor(small_shapes[1] / 2))  - (face_nose[1] + math.ceil(small_shapes[1] / 2)))
    underlay = face_image[face_nose[0] - math.floor(small_shapes[0] / 2):face_nose[0] + math.ceil(small_shapes[0] / 2), face_nose[1] - math.floor(small_shapes[1] / 2):face_nose[1] + math.ceil(small_shapes[1] / 2)]
    # cast = cropND(cast, underlay.shape)
    # cast = crop2DCenter(cast, underlay.shape, (np.array(cast.shape) / 2).astype(int))
    # SIO.imshow(underlay)
    # SIO.show()
    # SIO.imshow(cast)
    # SIO.show()
    first = np.copy(face_image[face_nose[0] - math.floor(small_shapes[0] / 2):face_nose[0] + math.ceil(small_shapes[0] / 2), face_nose[1] - math.floor(small_shapes[1] / 2):face_nose[1] + math.ceil(small_shapes[1] / 2)])

    # SIO.imshow(cast)
    # SIO.show()
    # face_image[face_nose[0] - math.floor(small_shapes[0] / 2):face_nose[0] + math.ceil(small_shapes[0] / 2), face_nose[1] - math.floor(small_shapes[1] / 2):face_nose[1] + math.ceil(small_shapes[1] / 2)] = cast
    face_image[max(face_nose[0] - math.floor(filter_half_ht), 0):min(face_nose[0] + math.ceil(filter_half_ht), face_image.shape[0]),max(face_nose[1] - math.floor(filter_half_wd), 0):min(face_nose[1] + math.ceil(filter_half_wd), face_image.shape[1])] = cast
    
    # underlay = cast
    # face_image = np.absolute(face_image)
    ########## END FAST ############

    # face_image = np.absolute(face_image)
    # # TODO: REPLACE THIS CODE
    # filter_over_spots = np.array(np.where(rotated_filter != 0))
    # for index in range(len(filter_over_spots[0])):
    #     point = [filter_over_spots[0][index], filter_over_spots[1][index]]
    #     displacement_from_center = [rotated_center[0] - point[0], rotated_center[1] - point[1]]
    #     try:
    #         face_image[face_nose[0] - displacement_from_center[0]][face_nose[1] - displacement_from_center[1]] = rotated_filter[point[0], point[1]]
    #         # print('success')
    #     except IndexError:
    #         # this will sometimes happen from resizing the filter so it's
    #         # too large; we want to just ignore these cases
    #         0

    # print(np.any(face_image < 0))
    # print(face_image.shape == orig_shape)
    # SIO.imshow(face_image)
    # SIO.show()
    # SIO.imshow(orig_face)
    # SIO.show()
    # SIO.imshow(first)
    # SIO.show()
    # SIO.imshow(cast)
    # SIO.show()
    # print(np.any(face_image < 0))
    # print(np.any(face_image > 255.))
    return face_image.astype(int)

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


def crop2DCenter(img, size, center):
    # print(img.shape)
    # print(size)
    # print(center)
    centerY, centerX = center
    sizeY, sizeX = size
    sizeYLower = math.floor(sizeY / 2)
    sizeYUpper = math.ceil(sizeY / 2)
    sizeXLower = math.floor(sizeX / 2)
    sizeXUpper = math.ceil(sizeX / 2)
    YLowerBound = centerY - sizeYLower
    YUpperBound = centerY + sizeYUpper
    XLowerBound = centerX - sizeXLower
    XUpperBound = centerX + sizeXUpper
    assert YLowerBound >= 0
    assert YUpperBound <= img.shape[0]
    assert XLowerBound >= 0
    assert XUpperBound <= img.shape[1]
    return img[YLowerBound:YUpperBound,XLowerBound:XUpperBound]
"""
from https://stackoverflow.com/a/50322574
"""
def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


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