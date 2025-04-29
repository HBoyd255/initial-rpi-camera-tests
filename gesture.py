import time
import numpy
from modules.eye import Eye

from modules.localiser import Localiser
from modules.fps import FPS
from modules.topDown import TopDown
from modules.video import Video
from modules.zoom import Zoom


from typing import NamedTuple, cast
from modules.hand import Hand

USE_THREADS = False

if USE_THREADS:
    from threading import Thread
    from queue import Queue
else:
    from multiprocessing import Process, Queue


class FrameStruct(NamedTuple):
    frame: numpy.ndarray
    hand: Hand


left_queue = Queue(maxsize=1)
right_queue = Queue(maxsize=1)

# Cast for type checking and autocomplete
left_queue = cast("Queue[FrameStruct]", left_queue)
right_queue = cast("Queue[FrameStruct]", right_queue)

localiser = Localiser()


fps = FPS()

top_down = TopDown(x_plot_bounds=(-0.25, 0.75), draw_grid=False)


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


GESTURES = [""] * 32
GESTURES[0b00000] = "Fist"
GESTURES[0b00001] = "Thumb"
GESTURES[0b00010] = "Index"
GESTURES[0b00011] = "German Two"
GESTURES[0b00100] = "Middle"
GESTURES[0b00101] = "Middle and Thumb"
GESTURES[0b00110] = "Peace"
GESTURES[0b00111] = "German Three"
GESTURES[0b01000] = "Ring"
GESTURES[0b01001] = "Ring and Thumb"
GESTURES[0b01010] = "Ring and Index"
GESTURES[0b01011] = "Ring, Index and Thumb"
GESTURES[0b01100] = "Ring and Middle"
GESTURES[0b01101] = "Ring and Middle and Thumb"
GESTURES[0b01110] = "American Three"
GESTURES[0b01111] = "German Four"
GESTURES[0b10000] = "Pinky"
GESTURES[0b10001] = "Call Me"
GESTURES[0b10010] = "Rock On"
GESTURES[0b10011] = "Spider-man"
GESTURES[0b10100] = "Middle and Pinky"
GESTURES[0b10101] = "Pinky, Middle and Thumb"
GESTURES[0b10110] = "Index, Middle and Pinky"
GESTURES[0b10111] = "Index, Middle, Pinky and Thumb"
GESTURES[0b11000] = "Pinky and Ring"
GESTURES[0b11001] = "Pinky, Ring and Thumb"
GESTURES[0b11010] = "Pinky, Ring and Index"
GESTURES[0b11011] = "Pinky, Ring, Index and Thumb"
GESTURES[0b11100] = "Pinky, Ring and Middle"
GESTURES[0b11101] = "Pinky, Ring, Middle and Thumb"
GESTURES[0b11110] = "American Four"
GESTURES[0b11111] = "Halt"


def finger_bend(A, B, C) -> float:

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


def is_pointed(points_3d: list, finger_index: int) -> bool:

    bend_thresholds = (120, 60, 60, 60, 60)

    wrist_indexes = (17, 0, 0, 0, 0)
    bend_indexes = (5, 6, 10, 14, 18)
    tip_indexes = (4, 8, 12, 16, 20)

    bend_threshold = bend_thresholds[finger_index]

    wrist = points_3d[wrist_indexes[finger_index]]
    knuckle = points_3d[bend_indexes[finger_index]]
    tip = points_3d[tip_indexes[finger_index]]

    bend_rads = finger_bend(wrist, knuckle, tip)

    bend_deg = numpy.rad2deg(bend_rads)

    finger_is_pointed = bend_deg > bend_threshold

    return finger_is_pointed


def get_gesture(hand_points):

    hand_points = numpy.array(hand_points)

    gesture_index = 0

    gesture_index |= is_pointed(hand_points, 0) << 0
    gesture_index |= is_pointed(hand_points, 1) << 1
    gesture_index |= is_pointed(hand_points, 2) << 2
    gesture_index |= is_pointed(hand_points, 3) << 3
    gesture_index |= is_pointed(hand_points, 4) << 4

    gesture = GESTURES[gesture_index]

    return gesture


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

    del left_feed, right_feed

    frame_an = numpy.copy(left_frame)

    if not (left_hand.is_seen() and right_hand.is_seen()):
        return

    hand_coords = localiser.get_coords(left_hand, right_hand)

    top_down.add_hand_points(hand_coords)
    frame_an = localiser.circle_3d_list(frame_an, hand_coords)

    gesture = get_gesture(hand_coords)

    print(gesture)

    vid.show("Projection", frame_an)

    top_down_image = top_down.get_image()

    vid.show("Top Down", top_down_image)


def show():

    vid = Video(canvas_framing=(2, 2))

    while True:

        main_loop(vid)


if __name__ == "__main__":

    if USE_THREADS:
        left_thread = Thread(target=capture_hand, args=("left", left_queue))
        right_thread = Thread(target=capture_hand, args=("right", right_queue))
        display_thread = Thread(target=show)

    # use process
    else:

        left_thread = Process(target=capture_hand, args=("left", left_queue))
        right_thread = Process(target=capture_hand, args=("right", right_queue))
        display_thread = Process(target=show)

    left_thread.start()
    right_thread.start()
    display_thread.start()

    left_thread.join()
    right_thread.join()
    display_thread.join()
