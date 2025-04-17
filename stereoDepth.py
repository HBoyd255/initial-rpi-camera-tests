from collections import namedtuple
import cv2
import numpy
import mediapipe
from modules.eye import Eye
from queue import Queue
from threading import Thread

from modules.distanceCalculator import DistanceCalculator
from modules.wheels import Wheels
from modules.video import Video
from modules.zoom import Zoom

mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles

FrameStruct = namedtuple("FrameStruct", ["frame", "hand"])

left_queue = Queue(maxsize=1)
right_queue = Queue(maxsize=1)

vid = Video()


wheel_control = Wheels()

BASELINE_M = 0.096
FOCAL_LENGTH_PX = 1964
FRAME_WIDTH_PX = 4608

HORIZONTAL_FOV_DEGREES = 102
VERTICAL_FOV_DEGREES = 67

distCalc = DistanceCalculator(
    BASELINE_M,
    FOCAL_LENGTH_PX,
    FRAME_WIDTH_PX,
    HORIZONTAL_FOV_DEGREES,
    VERTICAL_FOV_DEGREES,
)

def normalize_speeds(speeds: numpy.ndarray) -> numpy.ndarray:

    # Find the maximum absolute value
    max_value = numpy.max(numpy.abs(speeds))

    if max_value > 100:
        speeds = speeds / max_value * 100

        speeds = speeds.astype(int)

    return speeds


def capture_hand(side: str, queue: Queue):

    eye = Eye(side)

    hand_finder = Zoom()
    while True:

        frame = eye.array(res="full")

        hand = hand_finder.get_hand(frame)

        frame = frame[::4, ::4]

        queue.put(FrameStruct(frame, hand))


def show():

    while True:

        if not left_queue.empty() and not right_queue.empty():

            left_frame, left_hand = left_queue.get()
            right_frame, right_hand = right_queue.get()

            left_feed = numpy.copy(left_frame)
            right_feed = numpy.copy(right_frame)

            if left_hand.is_seen():
                left_feed = left_hand.draw(left_feed)
            if right_hand.is_seen():
                right_feed = right_hand.draw(right_feed)

            vid.show("Left Feed", left_feed)
            vid.show("Right Feed", right_feed)

            if not (left_hand and right_hand):
                continue

            distances = distCalc.get_distances(left_hand, right_hand)

            vid.show("Disparity", left_feed)

            print(f"Wrist = {distances[0]:.4}, Index = {distances[8]:.4}")

            print(f"Wrist = {distances[0]:.4}, Index = {distances[8]:.4}")

            canv = numpy.zeros_like(left_feed)

            cv2.putText(
                canv,
                f"Wrist = {distances[0]:.4}",
                (30, 100),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                canv,
                f"Index = {distances[8]:.4}",
                (30, 200),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (0, 0, 255),
                2,
            )

            vid.show("", canv)


#             vertical_error = wrist_distance - 100
#             vertical_error = vertical_error * 2
#
#             MAX_VAL = 100
#
#             if vertical_error > MAX_VAL:
#                 vertical_error = MAX_VAL
#
#             if vertical_error < -MAX_VAL:
#                 vertical_error = -MAX_VAL
#
#             horizontal_error = 0.5 - rx
#
#             horizontal_error *= 500
#
#             if horizontal_error > MAX_VAL:
#                 horizontal_error = MAX_VAL
#
#             if horizontal_error < -MAX_VAL:
#                 horizontal_error = -MAX_VAL
#
#             vertical_error = int(vertical_error)
#             horizontal_error = int(horizontal_error)
#
#             # if abs(horizontal_error) < 100:
#             r = numpy.random.randint(0, 100)
#
#             sine = -1
#
#             if horizontal_error > 0:
#                 sine = 1
#
#             print(f"r = {r}, " f"h = {horizontal_error} ", end="")
#
#             if abs(horizontal_error) > r:
#                 horizontal_error = 80 * sine
#             else:
#                 horizontal_error = 0
#
#             # -------------------
#
#             r = numpy.random.randint(0, 100)
#
#             sine = -1
#
#             if vertical_error > 0:
#                 sine = 1
#
#             print(f"r = {r}, " f"v = {vertical_error} ", end="")
#
#             if abs(vertical_error) > r:
#                 vertical_error = 100 * sine
#             else:
#                 vertical_error = 0
#
#             print(
#                 f"Wrist = {wrist_distance:.4}, "
#                 f"vertical_error = {vertical_error} "
#                 f"horizontal_error = {horizontal_error}"
#             )
#
#             speeds = numpy.zeros(4, dtype=int)
#
#             speeds += [
#                 vertical_error,
#                 vertical_error,
#                 vertical_error,
#                 vertical_error,
#             ]
#
#             speeds += [
#                 -horizontal_error,
#                 horizontal_error,
#                 -horizontal_error,
#                 horizontal_error,
#             ]
#
#             speeds = normalize_speeds(speeds)
#
#             wheel_control.send(*speeds)
#
#         cv2.waitKey(1)


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
