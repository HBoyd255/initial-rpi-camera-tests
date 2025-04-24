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

        pixel_coords = (self.landmarks * (width, height)).astype(int)

        if self._seen_from_zoom:
            base_colour = GREEN
        else:
            base_colour = RED

        width = 2

        for p in pixel_coords:
            cv2.circle(drawing_frame, p, 3, base_colour, -1)

        thumb = pixel_coords[1:5]
        index = pixel_coords[5:9]
        middle = pixel_coords[9:13]
        ring = pixel_coords[13:17]
        pinky = pixel_coords[17:21]

        palm = pixel_coords[[1, 0, 17, 0, 5]]
        knuckles = pixel_coords[[5, 9, 13, 17]]

        cv2.polylines(drawing_frame, [thumb], False, THUMB_COLOUR, width)
        cv2.polylines(drawing_frame, [index], False, INDEX_COLOUR, width)
        cv2.polylines(drawing_frame, [middle], False, MIDDLE_COLOUR, width)
        cv2.polylines(drawing_frame, [ring], False, RING_COLOUR, width)
        cv2.polylines(drawing_frame, [pinky], False, PINKY_COLOUR, width)

        cv2.polylines(
            drawing_frame, [palm, knuckles], False, PALM_COLOUR, width
        )

        for p in pixel_coords:
            cv2.circle(drawing_frame, p, 1, WHITE_RGB, -1)

        return drawing_frame
