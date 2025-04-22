import numpy
import cv2


aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
detector = cv2.aruco.ArucoDetector(aruco_dictionary)


class Aruco:

    def __init__(self, frame, seen_from_zoom=False):

        if frame is None:
            self._seen = False
            return

        corners, ids, _ = detector.detectMarkers(frame)

        self._seen = ids is not None

        if not self._seen:
            return

        height = len(frame)
        width = len(frame[0])

        self.landmarks = numpy.array(corners[0][0], dtype=float)

        self.landmarks /= [width, height]

        self._seen_from_zoom = seen_from_zoom

    def get_centre(self):

        if not self.is_seen():
            raise Exception("No tag seen")

        # In between the top left and bottom right corner.
        return (self.landmarks[0] + self.landmarks[2]) / 2

    def is_seen(self):
        return self._seen

    def draw(self, frame: numpy.ndarray):

        drawing_frame = numpy.copy(frame)

        if not self.is_seen():
            return drawing_frame

        height = len(drawing_frame)
        width = len(drawing_frame[0])

        pixel_coords = numpy.copy(self.landmarks)

        pixel_coords *= [width, height]

        pixel_coords = numpy.array(pixel_coords, dtype=int)

        colour = (0, 0, 255)

        if self._seen_from_zoom:
            colour = (0, 255, 0)

        cv2.circle(drawing_frame, pixel_coords[0], 3, colour, -1)
        cv2.circle(drawing_frame, pixel_coords[1], 3, colour, -1)
        cv2.circle(drawing_frame, pixel_coords[2], 3, colour, -1)
        cv2.circle(drawing_frame, pixel_coords[3], 3, colour, -1)

        point_list = [pixel_coords]

        cv2.polylines(
            drawing_frame, point_list, isClosed=True, color=(255, 0, 0)
        )

        return drawing_frame


# from eye import Eye
# from video import Video
#
#
# eye = Eye()
# vid = Video()
#
# while True:
#
#     frame = eye.array(res="full")
#
#     tag = Aruco(frame)
#
#     frame = numpy.array(frame[::4, ::4])
#
#     frame = tag.draw(frame)
#
#     vid.show("tag", frame)
