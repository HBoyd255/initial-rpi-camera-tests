import numpy
import numbers


class GestureClassifier:
    _GESTURE_NAMES = [""] * 32
    _GESTURE_NAMES[0b00000] = "Fist"
    _GESTURE_NAMES[0b00001] = "Thumb"
    _GESTURE_NAMES[0b00010] = "Index"
    _GESTURE_NAMES[0b00011] = "German Two"
    _GESTURE_NAMES[0b00100] = "Middle"
    _GESTURE_NAMES[0b00101] = "Middle and Thumb"
    _GESTURE_NAMES[0b00110] = "Peace"
    _GESTURE_NAMES[0b00111] = "German Three"
    _GESTURE_NAMES[0b01000] = "Ring"
    _GESTURE_NAMES[0b01001] = "Ring and Thumb"
    _GESTURE_NAMES[0b01010] = "Ring and Index"
    _GESTURE_NAMES[0b01011] = "Ring, Index and Thumb"
    _GESTURE_NAMES[0b01100] = "Ring and Middle"
    _GESTURE_NAMES[0b01101] = "Ring and Middle and Thumb"
    _GESTURE_NAMES[0b01110] = "American Three"
    _GESTURE_NAMES[0b01111] = "German Four"
    _GESTURE_NAMES[0b10000] = "Pinky"
    _GESTURE_NAMES[0b10001] = "Call Me"
    _GESTURE_NAMES[0b10010] = "Rock On"
    _GESTURE_NAMES[0b10011] = "Spider-Man"
    _GESTURE_NAMES[0b10100] = "Middle and Pinky"
    _GESTURE_NAMES[0b10101] = "Pinky, Middle and Thumb"
    _GESTURE_NAMES[0b10110] = "Index, Middle and Pinky"
    _GESTURE_NAMES[0b10111] = "Index, Middle, Pinky and Thumb"
    _GESTURE_NAMES[0b11000] = "Pinky and Ring"
    _GESTURE_NAMES[0b11001] = "Pinky, Ring and Thumb"
    _GESTURE_NAMES[0b11010] = "Pinky, Ring and Index"
    _GESTURE_NAMES[0b11011] = "Pinky, Ring, Index and Thumb"
    _GESTURE_NAMES[0b11100] = "Pinky, Ring and Middle"
    _GESTURE_NAMES[0b11101] = "Pinky, Ring, Middle and Thumb"
    _GESTURE_NAMES[0b11110] = "American Four"
    _GESTURE_NAMES[0b11111] = "Halt"

    def __init__(self):
        pass

    def _finger_bend(self, A, B, C) -> float:

        A = numpy.array(A)
        B = numpy.array(B)
        C = numpy.array(C)

        v1 = A - B
        v2 = C - B

        v1_mag = numpy.linalg.norm(v1)
        v1_norm = v1 / v1_mag

        v2_mag = numpy.linalg.norm(v2)
        v2_norm = v2 / v2_mag

        cos_theta = numpy.dot(v1_norm, v2_norm)

        cos_theta = numpy.clip(cos_theta, -1, 1)

        theta = numpy.arccos(cos_theta)

        return theta

    def _is_pointed(self, points_3d: list, finger_index: int) -> bool:

        bend_thresholds = (120, 60, 60, 60, 60)

        wrist_indexes = (17, 0, 0, 0, 0)
        bend_indexes = (5, 5, 9, 13, 17)
        tip_indexes = (4, 8, 12, 16, 20)

        bend_threshold = bend_thresholds[finger_index]

        wrist = points_3d[wrist_indexes[finger_index]]
        knuckle = points_3d[bend_indexes[finger_index]]
        tip = points_3d[tip_indexes[finger_index]]

        bend_rads = self._finger_bend(wrist, knuckle, tip)

        bend_deg = numpy.rad2deg(bend_rads)

        finger_is_pointed = bend_deg > bend_threshold

        return finger_is_pointed

    def get_gesture_id(self, hand_points: numpy.ndarray) -> int:

        hand_points = numpy.array(hand_points)

        gesture_index = 0

        gesture_index |= self._is_pointed(hand_points, 0) << 0
        gesture_index |= self._is_pointed(hand_points, 1) << 1
        gesture_index |= self._is_pointed(hand_points, 2) << 2
        gesture_index |= self._is_pointed(hand_points, 3) << 3
        gesture_index |= self._is_pointed(hand_points, 4) << 4

        return gesture_index

    def get_gesture_name(self, input_data) -> str:

        if isinstance(input_data, numbers.Integral):
            return self._GESTURE_NAMES[input_data]
        elif isinstance(input_data, (list, tuple, numpy.ndarray)):
            gesture_id = self.get_gesture_id(input_data)
            return self._GESTURE_NAMES[gesture_id]

        return "No Idea"
        # return self._GESTURE_NAMES[gesture_id]
