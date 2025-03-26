import numpy
import cv2


class Hand:

    def __init__(self, landmarks, is_left):

        self.is_left = is_left
        self.landmarks = landmarks

        self.angle_rad = 0.0
        self.gesture = "Unknown"

    def __str__(self):
        if self.is_left:
            return "Left Hand"
        return "Right Hand"

    def draw(self, frame: numpy.ndarray):

        height = len(frame)
        width = len(frame[0])

        pixel_coords = [
            (int(coord[0] * width), int(coord[1] * height))
            for coord in self.landmarks
        ]

        white = (255, 255, 255)
        black = (0, 0, 0)
        colour3 = (255, 0, 0)

        def line(p1_index: int, p2_index: int):
            cv2.line(
                frame,
                pixel_coords[p1_index],
                pixel_coords[p2_index],
                black,
                3,
            )
            cv2.line(
                frame,
                pixel_coords[p1_index],
                pixel_coords[p2_index],
                white,
                2,
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

        for x in range(21):
            cv2.circle(frame, pixel_coords[x], 4, white, -1)
            cv2.circle(frame, pixel_coords[x], 5, black, 1)


def create_hands_list(results):

    if not results.multi_hand_landmarks:
        return []

    hand_count = len(results.multi_hand_landmarks)

    is_left_list = [
        handedness.classification[0].label.casefold() == "left"
        for handedness in results.multi_handedness
    ]

    landmark_lists = results.multi_hand_landmarks

    hand_list = []

    for x in range(hand_count):

        landmarks = numpy.array(
            [
                (landmark.x, landmark.y)
                for landmark in landmark_lists[x].landmark
            ]
        )

        hand = Hand(landmarks, is_left_list[x])

        hand_list.append(hand)

    return hand_list
