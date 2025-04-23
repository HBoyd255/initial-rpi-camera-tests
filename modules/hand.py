import numpy
import cv2

from modules.colours import *


class Hand:

    def __init__(self, results, seen_from_zoom=False):

        if results is not None:
            self._seen = results.multi_hand_landmarks is not None

        else:
            self._seen = False

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

        self._seen_from_zoom = seen_from_zoom

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
        return (self.landmarks[0] + self.landmarks[9]) / 2

        # # Knuckle of the pointer finger
        # return self.landmarks[5]

        # Wrist
        # return self.landmarks[0]

    def draw(self, frame: numpy.ndarray):

        drawing_frame = numpy.copy(frame)

        if not self.is_seen():
            return drawing_frame

        height = len(drawing_frame)
        width = len(drawing_frame[0])

        pixel_coords = [
            (int(coord[0] * width), int(coord[1] * height))
            for coord in self.landmarks
        ]

        if self._seen_from_zoom:
            base_colour = GREEN_BGR
        else:
            base_colour = RED_BGR

        width = 2

        def line(p1_index: int, p2_index: int, colour: numpy.ndarray):

            cv2.line(
                drawing_frame,
                pixel_coords[p1_index],
                pixel_coords[p2_index],
                colour,
                width,
            )

        for p in pixel_coords:
            cv2.circle(drawing_frame, p, 3, base_colour, -1)

        # Thumb
        line(1, 2, THUMB_COLOUR_BGR)
        line(2, 3, THUMB_COLOUR_BGR)
        line(3, 4, THUMB_COLOUR_BGR)

        # Index
        line(5, 6, INDEX_COLOUR_BGR)
        line(6, 7, INDEX_COLOUR_BGR)
        line(7, 8, INDEX_COLOUR_BGR)

        # Middle
        line(9, 10, MIDDLE_COLOUR_BGR)
        line(10, 11, MIDDLE_COLOUR_BGR)
        line(11, 12, MIDDLE_COLOUR_BGR)

        # Ring
        line(13, 14, RING_COLOUR_BGR)
        line(14, 15, RING_COLOUR_BGR)
        line(15, 16, RING_COLOUR_BGR)

        # Ring
        line(17, 18, PINKY_COLOUR_BGR)
        line(18, 19, PINKY_COLOUR_BGR)
        line(19, 20, PINKY_COLOUR_BGR)

        # Wrist to knuckles
        line(0, 1, PALM_COLOUR_BGR)
        line(0, 5, PALM_COLOUR_BGR)
        line(0, 17, PALM_COLOUR_BGR)

        # Knuckles to each other
        line(5, 9, PALM_COLOUR_BGR)
        line(9, 13, PALM_COLOUR_BGR)
        line(13, 17, PALM_COLOUR_BGR)

        for p in pixel_coords:
            cv2.circle(drawing_frame, p, 1, WHITE, -1)

        return drawing_frame
