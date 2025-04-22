import cv2
import mediapipe
import numpy

mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
mp_pose = mediapipe.solutions.pose


class Body:

    def __init__(self, results):

        self._seen = results.pose_landmarks is not None

        if not self._seen:
            self.landmarks = numpy.zeros((33, 2), numpy.float64)
            return

        self.landmarks = numpy.array(
            [
                (landmark.x, landmark.y)
                for landmark in results.pose_landmarks.landmark
            ]
        )

    def is_seen(self):
        return self._seen

    def __str__(self):

        if not self.is_seen():
            return "No Body Seen."

        return "Body"

    def draw(self, frame: numpy.ndarray, small=False):

        drawing_frame = numpy.copy(frame)

        if not self.is_seen():
            return drawing_frame

        height = len(drawing_frame)
        width = len(drawing_frame[0])

        pixel_coords = [
            (int(coord[0] * width), int(coord[1] * height))
            for coord in self.landmarks
        ]

        colour = (255, 0, 255)

        if small:
            width = 1
        else:
            width = 3

        def line(p1_index: int, p2_index: int):
            cv2.line(
                drawing_frame,
                pixel_coords[p1_index],
                pixel_coords[p2_index],
                colour,
                width,
            )

        line(16, 14)
        line(14, 12)
        line(12, 11)
        line(11, 13)
        line(13, 15)

        line(28, 26)
        line(26, 24)
        line(24, 23)
        line(23, 25)
        line(25, 27)

        cv2.circle(drawing_frame, pixel_coords[16], 3, (0, 0, 255), -1)
        cv2.circle(drawing_frame, pixel_coords[14], 3, (0, 0, 255), -1)
        cv2.circle(drawing_frame, pixel_coords[12], 3, (0, 0, 255), -1)
        cv2.circle(drawing_frame, pixel_coords[11], 3, (0, 0, 255), -1)
        cv2.circle(drawing_frame, pixel_coords[13], 3, (0, 0, 255), -1)
        cv2.circle(drawing_frame, pixel_coords[15], 3, (0, 0, 255), -1)
        cv2.circle(drawing_frame, pixel_coords[28], 3, (0, 0, 255), -1)
        cv2.circle(drawing_frame, pixel_coords[26], 3, (0, 0, 255), -1)
        cv2.circle(drawing_frame, pixel_coords[24], 3, (0, 0, 255), -1)
        cv2.circle(drawing_frame, pixel_coords[23], 3, (0, 0, 255), -1)
        cv2.circle(drawing_frame, pixel_coords[25], 3, (0, 0, 255), -1)
        cv2.circle(drawing_frame, pixel_coords[27], 3, (0, 0, 255), -1)

        return drawing_frame
