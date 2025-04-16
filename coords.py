from collections import namedtuple
import cv2
import numpy
import mediapipe
from modules.eye import Eye
from queue import Queue
from threading import Thread

from modules.DistanceCalculator import DistanceCalculator
from modules.video import Video
from modules.zoom import Zoom

mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles

FrameStruct = namedtuple("FrameStruct", ["frame", "hand"])

left_queue = Queue(maxsize=1)
right_queue = Queue(maxsize=1)

vid = Video()

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

            left_feed = left_hand.draw(left_feed)
            right_feed = right_hand.draw(right_feed)

            vid.show("Left Feed", left_feed)
            vid.show("Right Feed", right_feed)

            
            if not (left_hand.is_seen() and right_hand.is_seen()):
                continue

            coords = distCalc.get_coords(left_hand, right_hand)

            print(coords[0])


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
