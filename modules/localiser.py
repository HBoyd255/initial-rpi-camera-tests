import numpy

from modules.translation import *

def camera_to_global(points):
    # Rotate 21 degrees around the x axis, then move 19.2cm up the Z axis

    ELEVATION_OFFSET_DEGREES = -21

    rotation_angle = numpy.radians(ELEVATION_OFFSET_DEGREES)

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


class Localiser:

    def __init__(
        self, baseline_m, focal_length_px, frame_width_px, h_fov_deg, v_fov_deg
    ):
        self._baseline_m = baseline_m
        self._focal_length_px = focal_length_px
        self._frame_width_px = frame_width_px

        self._distance_numerator = (
            self._baseline_m * self._focal_length_px / self._frame_width_px
        )

        self._h_fov_rad = numpy.radians(h_fov_deg)
        self._v_fov_rad = numpy.radians(v_fov_deg)

    def get_disparities(self, left_eye_hand, right_eye_hand):

        disparities = (
            left_eye_hand.landmarks[:, 0] - right_eye_hand.landmarks[:, 0]
        )

        return disparities

    def get_distances(self, left_eye_hand, right_eye_hand):

        disparities = self.get_disparities(left_eye_hand, right_eye_hand)

        distances = self._distance_numerator / disparities

        return distances

    def get_coords(self, left_eye_hand, right_eye_hand):

        landmarks = numpy.array(left_eye_hand.landmarks)
        distances = numpy.array(
            self.get_distances(left_eye_hand, right_eye_hand)
        )

        azimuth_angles_rad = (landmarks[:, 0] - 0.5) * self._h_fov_rad
        elevation_angles_rad = (0.5 - landmarks[:, 1]) * self._v_fov_rad

        x_vals = (
            distances
            * numpy.cos(elevation_angles_rad)
            * numpy.sin(azimuth_angles_rad)
        )
        y_vals = (
            distances
            * numpy.cos(elevation_angles_rad)
            * numpy.cos(azimuth_angles_rad)
        )

        z_vals = distances * numpy.sin(elevation_angles_rad)

        cam_relative = numpy.column_stack([x_vals, y_vals, z_vals])

        global_coords = camera_to_global(cam_relative)

        return global_coords

