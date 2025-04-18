


import cv2
import numpy

HORIZONTAL_FOV_DEGREES = 102
VERTICAL_FOV_DEGREES = 67

def camera_to_global(points):
    # Rotate 21 degrees around the x axis, then move 19.2cm up the Z axis

    AZIMUTH_OFFSET_DEGREES = -21

    rotation_angle = numpy.radians(AZIMUTH_OFFSET_DEGREES)

    rotation_matrix = numpy.array(
        [
            [1, 0, 0],
            [0, numpy.cos(rotation_angle), -numpy.sin(rotation_angle)],
            [0, numpy.sin(rotation_angle), numpy.cos(rotation_angle)],
        ]
    )

    rotated_points = points @ rotation_matrix

    rotated_points += [0, 0, 0.192]

    return rotated_points


def global_to_camera(points):

    transformed_points = numpy.copy(points)

    transformed_points -= [0, 0, 0.192]

    AZIMUTH_OFFSET_DEGREES = 21

    rotation_angle = numpy.radians(AZIMUTH_OFFSET_DEGREES)

    rotation_matrix = numpy.array(
        [
            [1, 0, 0],
            [0, numpy.cos(rotation_angle), -numpy.sin(rotation_angle)],
            [0, numpy.sin(rotation_angle), numpy.cos(rotation_angle)],
        ]
    )

    transformed_points = transformed_points @ rotation_matrix

    return transformed_points


def circle_2d(frame, coord):

    drawing_frame = numpy.copy(frame)

    height = len(frame)
    width = len(frame[0])

    p_x = int(coord[0] * width)
    p_y = int(coord[1] * height)

    cv2.circle(drawing_frame, (p_x, p_y), 3, (255, 0, 0), -1)

    return drawing_frame


def circle_3d(frame, coord_3d):

    drawing_frame = numpy.copy(frame)

    x = coord_3d[0]
    y = coord_3d[1]
    z = coord_3d[2]

    e = numpy.arctan2(z, y)
    a = numpy.arctan2(x, y)

    X_2 = (a / numpy.radians(HORIZONTAL_FOV_DEGREES)) + 0.5
    Y_2 = 0.5 - (e / numpy.radians(VERTICAL_FOV_DEGREES))

    drawing_frame = circle_2d(drawing_frame, (X_2, Y_2))

    return drawing_frame


def line_2d(frame, coord1, coord2, colour=(255, 0, 255)):

    drawing_frame = numpy.copy(frame)

    height = len(frame)
    width = len(frame[0])

    p1 = (int(coord1[0] * width), int(coord1[1] * height))
    p2 = (int(coord2[0] * width), int(coord2[1] * height))

    cv2.line(drawing_frame, p1, p2, colour, 3)

    return drawing_frame


def line_3d(frame, coord1_3d, coord2_3d, colour=((255, 0, 255))):

    drawing_frame = numpy.copy(frame)

    x1 = coord1_3d[0]
    y1 = coord1_3d[1]
    z1 = coord1_3d[2]

    e1 = numpy.arctan2(z1, y1)
    a1 = numpy.arctan2(x1, y1)

    X1 = (a1 / numpy.radians(HORIZONTAL_FOV_DEGREES)) + 0.5
    Y1 = 0.5 - (e1 / numpy.radians(VERTICAL_FOV_DEGREES))

    x2 = coord2_3d[0]
    y2 = coord2_3d[1]
    z2 = coord2_3d[2]

    e2 = numpy.arctan2(z2, y2)
    a2 = numpy.arctan2(x2, y2)

    X2 = (a2 / numpy.radians(HORIZONTAL_FOV_DEGREES)) + 0.5
    Y2 = 0.5 - (e2 / numpy.radians(VERTICAL_FOV_DEGREES))

    # print(coord1_3d,(X1, Y1), (X2, Y2))
    # print(a1, a2)
    # print((x1, y1), a1)

    drawing_frame = line_2d(drawing_frame, (X1, Y1), (X2, Y2), colour=colour)

    return drawing_frame
