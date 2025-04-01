import cv2
import numpy


class StereoHandPair:

    def __init__(self, left_eye_hand, right_eye_hand):

        self.left_eye_hand = left_eye_hand
        self.right_eye_hand = right_eye_hand

    def draw(self, frame: numpy.ndarray):

        height = len(frame)
        width = len(frame[0])

        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        WHITE = (255, 255, 255)

        l_pixel_coords = [
            (int(coord[0] * width), int(coord[1] * height))
            for coord in self.left_eye_hand.landmarks
        ]

        r_pixel_coords = [
            (int(coord[0] * width), int(coord[1] * height))
            for coord in self.right_eye_hand.landmarks
        ]

        for x in range(21):
            cv2.line(frame, l_pixel_coords[x], r_pixel_coords[x], WHITE, 2)
            cv2.circle(frame, l_pixel_coords[x], 4, BLUE, -1)
            cv2.circle(frame, r_pixel_coords[x], 4, RED, -1)

    def get_disparities(self):

        disparities = [
            self.left_eye_hand.landmarks[x][0]
            - self.right_eye_hand.landmarks[x][0]
            for x in range(21)
        ]

        return disparities
