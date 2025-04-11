import numpy
import cv2


class Hand:

    def __init__(self, results):

        self._seen = results.multi_hand_landmarks is not None

        if not self._seen:
            self._is_left = False
            self.landmarks = numpy.zeros((21, 2), numpy.float64)

            return

        self._is_left = (
            results.multi_handedness[0].classification[0].label.casefold()
            == "left"
        )

        self.landmarks = numpy.array(
            [
                (landmark.x, landmark.y)
                for landmark in results.multi_hand_landmarks[0].landmark
            ]
        )

        # self.angle_rad = 0.0
        # self.gesture = "Unknown"

    def is_seen(self):
        return self._seen

    def __str__(self):

        if not self.is_seen():
            return "No Hand Seen."

        if self._is_left:
            return "Left Hand"
        return "Right Hand"

    def get_centre(self):

        if not self.is_seen():
            raise Exception("No hands seen")

        # In between the wrist and the index knuckle.
        # return (self.landmarks[0] + self.landmarks[9]) / 2

        # # Knuckle of the pointer finger
        # return self.landmarks[5]

        # Wrist
        return self.landmarks[0]

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

        RED = (0, 0, 255)

        if small:
            width = 1
        else:
            width = 3

        def line(p1_index: int, p2_index: int):
            cv2.line(
                drawing_frame,
                pixel_coords[p1_index],
                pixel_coords[p2_index],
                RED,
                width,
            )

        # Fingers
        for x in range(3):
            for finger in range(5):
                line((finger * 4) + x + 1, (finger * 4) + x + 2)
        # Wrist to knuckles
        line(0, 1)
        line(0, 5)
        line(0, 17)

        # Knuckles to each other
        line(5, 9)
        line(9, 13)
        line(13, 17)

        return drawing_frame
