from collections import namedtuple
import cv2
import numpy
import mediapipe
from modules.eye import Eye
from modules.fps import FPS
from modules.hand import Hand
from queue import Queue
from threading import Thread

from modules.stereoHands import StereoHandPair
from modules.wheels import Wheels
from modules.video import Video

mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles

FrameStruct = namedtuple("FrameStruct", ["frame", "hand"])

left_queue = Queue(maxsize=1)
right_queue = Queue(maxsize=1)

vid = Video()

CAMERA_SPACING = 0.048
FOCAL_LENGTH = 0.00275

K = 1.9


wheel_control = Wheels()


def normalize_speeds(speeds: numpy.ndarray) -> numpy.ndarray:

    # Find the maximum absolute value
    max_value = numpy.max(numpy.abs(speeds))

    if max_value > 100:
        speeds = speeds / max_value * 100

        speeds = speeds.astype(int)

    return speeds


def capture_hand(side: str, queue: Queue):

    eye = Eye(side)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    while True:

        frame = eye.array("RGB")
        results = hands.process(frame)

        detected_hand = Hand(results)

        if not detected_hand.is_seen():
            detected_hand = None

        queue.put(FrameStruct(frame, detected_hand))


def show():

    fps = FPS()

    while True:

        if not left_queue.empty() and not right_queue.empty():

            left_frame, left_hand = left_queue.get()
            right_frame, right_hand = right_queue.get()

            left_feed = numpy.copy(left_frame)
            right_feed = numpy.copy(right_frame)

            if left_hand:
                left_hand.draw(left_feed)
            if right_hand:
                right_hand.draw(right_feed)

            BGR_frame = cv2.cvtColor(left_feed, cv2.COLOR_BGR2RGB)
            vid.show("Left Feed", BGR_frame)
            BGR_frame = cv2.cvtColor(right_feed, cv2.COLOR_BGR2RGB)
            vid.show("Right Feed", BGR_frame)

            if not (left_hand and right_hand):
                continue

            pair = StereoHandPair(left_hand, right_hand)

            pair.draw(left_feed)

            disparities = pair.get_disparities()

            distances = [K / d for d in disparities]

            BGR_frame = cv2.cvtColor(left_feed, cv2.COLOR_BGR2RGB)
            vid.show("Disparity", BGR_frame)

            lx = left_hand.landmarks[8][0]
            ly = left_hand.landmarks[0][1]

            rx = right_hand.landmarks[8][0]
            ry = right_hand.landmarks[0][1]

            # if ly < 0.5 and ry < 0.5:
            #     continue

            wrist_distance = distances[0]

            # print(f"Wrist = {wrist_distance:.4}, Index = {distances[8]:.4}")

            vertical_error = wrist_distance - 100
            vertical_error = vertical_error * 2

            MAX_VAL = 100

            if vertical_error > MAX_VAL:
                vertical_error = MAX_VAL

            if vertical_error < -MAX_VAL:
                vertical_error = -MAX_VAL

            horizontal_error = 0.5 - rx

            horizontal_error *= 500

            if horizontal_error > MAX_VAL:
                horizontal_error = MAX_VAL

            if horizontal_error < -MAX_VAL:
                horizontal_error = -MAX_VAL

            vertical_error = int(vertical_error)
            horizontal_error = int(horizontal_error)

            # if abs(horizontal_error) < 100:
            r = numpy.random.randint(0, 100)

            sine = -1

            if horizontal_error > 0:
                sine = 1

            print(f"r = {r}, " f"h = {horizontal_error} ", end="")

            if abs(horizontal_error) > r:
                horizontal_error = 80 * sine
            else:
                horizontal_error = 0

            # -------------------

            r = numpy.random.randint(0, 100)

            sine = -1

            if vertical_error > 0:
                sine = 1

            print(f"r = {r}, " f"v = {vertical_error} ", end="")

            if abs(vertical_error) > r:
                vertical_error = 100 * sine
            else:
                vertical_error = 0

            print(
                f"Wrist = {wrist_distance:.4}, "
                f"vertical_error = {vertical_error} "
                f"horizontal_error = {horizontal_error}"
            )

            speeds = numpy.zeros(4, dtype=int)

            speeds += [
                vertical_error,
                vertical_error,
                vertical_error,
                vertical_error,
            ]

            speeds += [
                -horizontal_error,
                horizontal_error,
                -horizontal_error,
                horizontal_error,
            ]

            speeds = normalize_speeds(speeds)

            wheel_control.send(*speeds)

        cv2.waitKey(1)


if __name__ == "__main__":

    left_thread = Thread(target=capture_hand, args=("left", left_queue))
    right_thread = Thread(target=capture_hand, args=("right", right_queue))

    display_thread = Thread(target=show)

    left_thread.start()
    right_thread.start()
    display_thread.start()

    left_thread.join()
    right_thread.join()
    display_thread.join()
