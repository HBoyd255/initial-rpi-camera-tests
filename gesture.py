import time
import numpy
from modules.eye import Eye

from modules.fps import FPS
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
left_queue = cast("Queue[FrameStruct]", left_queue)


fps = FPS()


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

    AB = B - A
    BC = C - B

    angle_AB = numpy.arctan2(*AB)
    angle_BC = numpy.arctan2(*BC)

    angle_diff = angle_BC - angle_AB  # Radians

    # Normalize the angle to be between -pi and pi
    angle_diff = numpy.mod(angle_diff + numpy.pi, 2 * numpy.pi) - numpy.pi

    degrees = numpy.degrees(angle_diff)

    return abs(degrees)


def is_pointed(landmarks: list, finger_index: int) -> bool:

    bend_threshold = 60

    bend_indexes = [2, 6, 10, 14, 18]
    tip_indexes = [4, 8, 12, 16, 20]

    wrist = landmarks[0]
    knuckle = landmarks[bend_indexes[finger_index]]
    tip = landmarks[tip_indexes[finger_index]]

    bend = finger_bend(wrist, knuckle, tip)

    return bend <= bend_threshold


def get_gesture(hand: Hand):

    gesture_index = 0

    gesture_index |= is_pointed(hand.landmarks, 0) << 0
    gesture_index |= is_pointed(hand.landmarks, 1) << 1
    gesture_index |= is_pointed(hand.landmarks, 2) << 2
    gesture_index |= is_pointed(hand.landmarks, 3) << 3
    gesture_index |= is_pointed(hand.landmarks, 4) << 4

    gesture = GESTURES[gesture_index]

    return gesture


def main_loop(vid: Video):
    if left_queue.empty():
        return

    left_frame, left_hand = left_queue.get()

    left_feed = numpy.copy(left_frame)

    left_feed = left_hand.draw(left_feed)

    gesture = get_gesture(left_hand)

    print(gesture)

    vid.show("Left Feed", left_feed)


def show():

    vid = Video(canvas_framing=(2, 2))

    while True:

        main_loop(vid)


if __name__ == "__main__":

    if USE_THREADS:
        left_thread = Thread(target=capture_hand, args=("left", left_queue))
        display_thread = Thread(target=show)

    # use process
    else:

        left_thread = Process(target=capture_hand, args=("left", left_queue))
        display_thread = Process(target=show)

    left_thread.start()
    display_thread.start()

    left_thread.join()
    display_thread.join()
