import time
from typing import NamedTuple, cast
import cv2
import numpy
import mediapipe
from modules.eye import Eye
from multiprocessing import Process, Queue

from modules.hand import Hand
from modules.localiser import Localiser
from modules.wheels import Wheels
from modules.video import Video
from modules.zoomLive import Zoom

mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles


class FrameStruct(NamedTuple):
    frame: numpy.ndarray
    hand: Hand


left_queue = Queue(maxsize=1)
right_queue = Queue(maxsize=1)

# Cast for type checking and autocomplete
left_queue = cast("Queue[FrameStruct]", left_queue)
right_queue = cast("Queue[FrameStruct]", right_queue)


wheel_control = Wheels()


localiser = Localiser()


def probabilistic(power):

    sine = -1
    if power > 0:
        sine = 1

    trip_power = 80

    rand = numpy.random.randint(0, trip_power)

    if abs(power) > rand:
        return trip_power * sine

    return 0


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

        if queue.full():
            time.sleep(0.01)

        frame = eye.array(res="full")

        hand = hand_finder.get_hand(frame, simple=False)

        frame = numpy.array(frame[::4, ::4])

        frame = hand_finder.draw_zoom_outline(frame)

        queue.put(FrameStruct(frame, hand))


def main_loop(vid: Video):
    if left_queue.empty() or right_queue.empty():
        return

    left_frame, left_hand = left_queue.get()
    right_frame, right_hand = right_queue.get()

    left_feed = numpy.copy(left_frame)
    right_feed = numpy.copy(right_frame)

    left_feed = left_hand.draw(left_feed)
    right_feed = right_hand.draw(right_feed)

    vid.show("Left Feed", left_feed)
    vid.show("Right Feed", right_feed)

    if not (left_hand.is_seen() and right_hand.is_seen()):
        return

    distances = localiser.get_distances(left_hand, right_hand)

    wrist_distance = distances[0]

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

    rx = left_hand.landmarks[0][0]

    vertical_error = (wrist_distance - 1) * 2
    horizontal_error = (0.5 - rx) * 5

    vertical_error = int(vertical_error * 100)
    horizontal_error = int(horizontal_error * 100)

    vertical_error = probabilistic(vertical_error)
    horizontal_error = probabilistic(horizontal_error)
    #
    #     print(
    #         f"Wrist = {wrist_distance:.4}, "
    #         f"vertical_error = {vertical_error} "
    #         f"horizontal_error = {horizontal_error}"
    #     )

    speeds = numpy.zeros(4, dtype=int)

    speeds += (
        vertical_error,
        vertical_error,
        vertical_error,
        vertical_error,
    )

    speeds += (
        -horizontal_error,
        horizontal_error,
        -horizontal_error,
        horizontal_error,
    )

    speeds = normalize_speeds(speeds)

    wheel_control.send(*speeds)


def show():

    vid = Video()

    while True:

        main_loop(vid)


if __name__ == "__main__":

    left_thread = Process(target=capture_hand, args=("left", left_queue))
    right_thread = Process(target=capture_hand, args=("right", right_queue))

    display_thread = Process(target=show)

    left_thread.start()
    right_thread.start()
    display_thread.start()

    left_thread.join()
    right_thread.join()
    display_thread.join()
