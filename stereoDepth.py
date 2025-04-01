from collections import namedtuple
import cv2
import numpy
import mediapipe
from modules.eye import Eye
from modules.fps import FPS
from modules.hand import create_hands_list
from queue import Queue
from threading import Thread

from modules.stereoHands import StereoHandPair

mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles

FrameStruct = namedtuple("FrameStruct", ["frame", "hand"])

left_queue = Queue(maxsize=1)
right_queue = Queue(maxsize=1)


CAMERA_SPACING = 0.048
FOCAL_LENGTH = 0.00275

K = 1.9


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
        hands_list = create_hands_list(results)

        detected_hand = None

        if len(hands_list) > 0:
            detected_hand = hands_list[0]

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
            cv2.imshow("Left Feed", BGR_frame)
            BGR_frame = cv2.cvtColor(right_feed, cv2.COLOR_BGR2RGB)
            cv2.imshow("Right Feed", BGR_frame)

            if not (left_hand and right_hand):
                continue

            pair = StereoHandPair(left_hand, right_hand)

            pair.draw(left_feed)

            disparities = pair.get_disparities()

            distances = [K / d for d in disparities]

            print(f"Wrist = {distances[0]:.4}, Index = {distances[8]:.4}")

            BGR_frame = cv2.cvtColor(left_feed, cv2.COLOR_BGR2RGB)
            cv2.imshow("Disparity", BGR_frame)

            fps.tick()

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
