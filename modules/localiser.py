import cv2
import numpy


class Localiser:

    def __init__(
        self,
        baseline_m,
        focal_length_px,
        frame_width_px,
        h_fov_deg,
        v_fov_deg,
        elevation_offset_deg,
        horizontal_offset_m,
    ):
        self._distance_numerator = baseline_m * focal_length_px / frame_width_px

        self._h_fov_rad = numpy.radians(h_fov_deg)
        self._v_fov_rad = numpy.radians(v_fov_deg)

        self._elevation_offset_rad = numpy.radians(elevation_offset_deg)
        self._horizontal_offset_m = horizontal_offset_m

    def get_disparities(self, left_eye_hand, right_eye_hand):

        disparities = (
            left_eye_hand.landmarks[:, 0] - right_eye_hand.landmarks[:, 0]
        )

        return disparities

    def get_distances(self, left_eye_hand, right_eye_hand):

        disparities = self.get_disparities(left_eye_hand, right_eye_hand)

        distances = self._distance_numerator / disparities

        return distances

    def get_coords(self, left_hand, right_hand):

        landmarks = numpy.array(left_hand.landmarks)
        distances = numpy.array(self.get_distances(left_hand, right_hand))

        # Array to the azimuth and elevation angles of each point in the hand,
        # measured in radians.
        azimuths = (landmarks[:, 0] - 0.5) * self._h_fov_rad
        elevations = (0.5 - landmarks[:, 1]) * self._v_fov_rad

        # Offset the elevations angels by the vertical angle of the camera.
        elevations -= self._elevation_offset_rad

        # Calculate the X, Y and Z coordinates of each point, in meters.
        x_vals = distances * numpy.cos(elevations) * numpy.sin(azimuths)
        y_vals = distances * numpy.cos(elevations) * numpy.cos(azimuths)
        z_vals = distances * numpy.sin(elevations)

        z_vals += self._horizontal_offset_m

        hand_coords = numpy.column_stack([x_vals, y_vals, z_vals])

        return hand_coords

    def circle_2d(self, frame, coord):

        drawing_frame = numpy.copy(frame)

        height = len(frame)
        width = len(frame[0])

        p_x = int(coord[0] * width)
        p_y = int(coord[1] * height)

        cv2.circle(drawing_frame, (p_x, p_y), 3, (255, 0, 0), -1)

        return drawing_frame

    def circle_3d(self, frame, coord_3d):

        drawing_frame = numpy.copy(frame)

        x = coord_3d[0]
        y = coord_3d[1]
        z = coord_3d[2]

        z -= self._horizontal_offset_m

        e = numpy.arctan2(z, y) + self._elevation_offset_rad
        a = numpy.arctan2(x, y)

        X = (a / self._h_fov_rad) + 0.5
        Y = 0.5 - (e / self._v_fov_rad)

        drawing_frame = self.circle_2d(drawing_frame, (X, Y))

        return drawing_frame

    def line_2d(self, frame, coord1, coord2, colour=(255, 0, 255)):

        drawing_frame = numpy.copy(frame)

        height = len(frame)
        width = len(frame[0])

        p1 = (int(coord1[0] * width), int(coord1[1] * height))
        p2 = (int(coord2[0] * width), int(coord2[1] * height))

        cv2.line(drawing_frame, p1, p2, colour, 3)

        return drawing_frame

    def line_3d(self, frame, coord1_3d, coord2_3d, colour=((255, 0, 255))):

        # coord2_3d = global_to_camera(coord2_3d)

        drawing_frame = numpy.copy(frame)

        x1 = coord1_3d[0]
        y1 = coord1_3d[1]
        z1 = coord1_3d[2]

        z1 -= self._horizontal_offset_m

        e1 = numpy.arctan2(z1, y1) + self._elevation_offset_rad
        a1 = numpy.arctan2(x1, y1)

        X1 = (a1 / self._h_fov_rad) + 0.5
        Y1 = 0.5 - (e1 / self._v_fov_rad)

        x2 = coord2_3d[0]
        y2 = coord2_3d[1]
        z2 = coord2_3d[2]

        z2 -= self._horizontal_offset_m

        e2 = numpy.arctan2(z2, y2) + self._elevation_offset_rad
        a2 = numpy.arctan2(x2, y2)

        X2 = (a2 / self._h_fov_rad) + 0.5
        Y2 = 0.5 - (e2 / self._v_fov_rad)

        drawing_frame = self.line_2d(
            drawing_frame, (X1, Y1), (X2, Y2), colour=colour
        )

        return drawing_frame
