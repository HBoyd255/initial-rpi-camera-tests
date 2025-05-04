import cv2
import numpy

from modules.colours import *
from modules.evaluateVariable import evaluate_variable
from modules.hand import Hand
from modules.physical import FOCAL_LENGTH_MM, SENSOR_SIZE, STEREO_BASELINE_M


def camera_to_global(points):
    # Rotate 21 degrees around the x axis, then move 19.2cm up the Z axis

    ELEVATION_OFFSET_DEGREES = 21

    rotation_angle = numpy.deg2rad(-ELEVATION_OFFSET_DEGREES)

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

    ELEVATION_OFFSET_DEGREES = 21

    rotation_angle = numpy.deg2rad(ELEVATION_OFFSET_DEGREES)

    rotation_matrix = numpy.array(
        [
            [1, 0, 0],
            [0, numpy.cos(rotation_angle), -numpy.sin(rotation_angle)],
            [0, numpy.sin(rotation_angle), numpy.cos(rotation_angle)],
        ]
    )

    transformed_points = transformed_points @ rotation_matrix

    return transformed_points


class Localiser:

    def __init__(
        self,
    ):
        self._distance_numerator = (
            FOCAL_LENGTH_MM / SENSOR_SIZE[0] * STEREO_BASELINE_M
        )

    def get_disparities(self, left_eye_hand, right_eye_hand):

        disparities = (
            left_eye_hand.landmarks[:, 0] - right_eye_hand.landmarks[:, 0]
        )

        return disparities

    def get_distances(self, left_eye_hand, right_eye_hand):

        disparities = self.get_disparities(left_eye_hand, right_eye_hand)

        distances = self._distance_numerator / disparities

        return distances

    def get_coords(self, left_hand: Hand, right_hand: Hand):

        # Get the coordinates using the pinhole method

        if not left_hand.is_seen() or not right_hand.is_seen():
            coords_global = numpy.full((21, 3), numpy.nan)
            return coords_global

        landmarks = numpy.array(left_hand.landmarks)
        distances = numpy.array(self.get_distances(left_hand, right_hand))

        # Center the landmarks
        landmarks -= [0.5, 0.5]

        landmarks[:, 1] = 0 - landmarks[:, 1]

        sensor_landmarks = SENSOR_SIZE * landmarks

        x_vals = sensor_landmarks[:, 0] * distances / FOCAL_LENGTH_MM
        y_vals = distances
        z_vals = sensor_landmarks[:, 1] * distances / FOCAL_LENGTH_MM

        coords_cam = numpy.column_stack((x_vals, y_vals, z_vals))

        coords_global = camera_to_global(coords_cam)

        return coords_global

    def _3d_to_landmark(self, coord_3d):

        cam_coords = global_to_camera(coord_3d)

        x_val = cam_coords[0]
        y_val = cam_coords[1]
        z_val = cam_coords[2]

        sensor_x = x_val / y_val * FOCAL_LENGTH_MM
        sensor_y = z_val / y_val * FOCAL_LENGTH_MM

        sensor_y = 0 - sensor_y

        sensor_landmark = numpy.array([sensor_x, sensor_y])

        landmark = sensor_landmark / SENSOR_SIZE

        landmark += [0.5, 0.5]

        return landmark

    def _3d_to_landmark_list(self, coord_3d_list):

        cam_coords = global_to_camera(coord_3d_list)

        x_vals = cam_coords[:, 0]
        y_vals = cam_coords[:, 1]
        z_vals = cam_coords[:, 2]

        sensor_x = (x_vals / y_vals) * FOCAL_LENGTH_MM
        sensor_y = (z_vals / y_vals) * FOCAL_LENGTH_MM

        sensor_y = 0 - sensor_y

        sensor_landmark = numpy.column_stack([sensor_x, sensor_y])

        landmarks = sensor_landmark / SENSOR_SIZE

        landmarks += [0.5, 0.5]

        return landmarks

    def circle_3d(self, frame, coord_3d, colour=BLUE):

        coord_3d = numpy.array(coord_3d, dtype=float)

        drawing_frame = numpy.copy(frame)

        height = len(drawing_frame)
        width = len(drawing_frame[0])

        if coord_3d[1] < 0:
            return drawing_frame

        landmark = self._3d_to_landmark(coord_3d)

        pixel_coords = numpy.array(landmark * [width, height], dtype=int)

        cv2.circle(drawing_frame, pixel_coords, 3, colour, -1)

        return drawing_frame

    def circle_3d_list(self, frame, coord_list, colour=BLUE):

        drawing_frame = numpy.copy(frame)

        height = len(drawing_frame)
        width = len(drawing_frame[0])

        # TODO add removal of negative y values.
        # if coord_list[1] < 0:
        #     return drawing_frame

        landmarks = self._3d_to_landmark_list(coord_list)

        landmarks_x = landmarks[:, 0]
        landmarks_y = landmarks[:, 1]

        pixels_x = (landmarks_x * width).astype(int)
        pixels_y = (landmarks_y * height).astype(int)

        pixel_coords = numpy.column_stack([pixels_x, pixels_y])

        for p in pixel_coords:

            cv2.circle(drawing_frame, p, 3, colour, -1)

        return drawing_frame

    def line_3d(self, frame, coord_3d, colour=MAGENTA):

        drawing_frame = numpy.copy(frame)

        coord_3d = numpy.array(coord_3d)

        height = len(drawing_frame)
        width = len(drawing_frame[0])

        if numpy.min(coord_3d[:, 1]) < 0:
            return drawing_frame

        landmarks = self._3d_to_landmark_list(coord_3d)

        pixel_coords = (landmarks * (width, height)).astype(int)

        cv2.polylines(
            drawing_frame, [pixel_coords], isClosed=False, color=colour
        )

        return drawing_frame
