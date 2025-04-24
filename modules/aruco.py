import numpy
import cv2

from modules.colours import *


class Aruco:

    def __init__(self, id, landmarks, seen_from_zoom=False):

        self.id = id
        self.landmarks = landmarks

        self._seen_from_zoom = seen_from_zoom

    def get_centre(self):

        # In between the top left and bottom right corner.
        return (self.landmarks[0] + self.landmarks[2]) / 2

    def draw(self, frame: numpy.ndarray):

        drawing_frame = numpy.copy(frame)

        height = len(drawing_frame)
        width = len(drawing_frame[0])

        pixel_coords = numpy.copy(self.landmarks)

        pixel_coords *= [width, height]

        pixel_coords = numpy.array(pixel_coords, dtype=int)

        if self._seen_from_zoom:
            base_colour = GREEN
        else:
            base_colour = RED

        for p in pixel_coords:
            cv2.circle(drawing_frame, p, 3, base_colour, -1)

        point_list = [pixel_coords]

        cv2.polylines(drawing_frame, point_list, isClosed=True, color=BLUE)

        return drawing_frame

    def __str__(self):

        return f"Aruco tag: {self.id}"


aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
detector = cv2.aruco.ArucoDetector(aruco_dictionary)


def aruco_list(frame, seen_from_zoom=False):

    list_of_aruco = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return list_of_aruco

    height = len(frame)
    width = len(frame[0])

    normalised_ids = numpy.array(ids)[:, 0]

    normalised_corners = numpy.array(corners, dtype=float)

    landmarks_list = normalised_corners[:, 0, :, :] / [width, height]

    sort_order = numpy.argsort(normalised_ids)

    sorted_ids = normalised_ids[sort_order]
    sorted_landmark_list = landmarks_list[sort_order]

    for i, id in enumerate(sorted_ids):

        new_aruco = Aruco(
            id, sorted_landmark_list[i], seen_from_zoom=seen_from_zoom
        )

        list_of_aruco.append(new_aruco)

    return list_of_aruco
