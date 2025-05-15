from collections import deque
import time
import numpy
from modules.colours import *
from modules.evaluateVariable import evaluate_variable
from modules.eye import Eye

from modules.localiser import Localiser
from modules.fps import FPS
from modules.topDown import TopDown
from modules.video import Video
from modules.zoomLive import Zoom


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

top_down = TopDown(x_plot_bounds=(-0.25, 4), draw_grid=True)


history = deque(maxlen=10)


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


def draw_square_on_ground(frame, ground_coord):

    drawing_frame = numpy.copy(frame)

    ground_coord = numpy.array(ground_coord)

    point1 = ground_coord + [0.1, 0.1, 0]
    point2 = ground_coord + [-0.1, 0.1, 0]
    point3 = ground_coord + [-0.1, -0.1, 0]
    point4 = ground_coord + [0.1, -0.1, 0]

    point_list = [point1, point2, point3, point4]

    drawing_frame = localiser.line_3d(drawing_frame, point_list)

    point_list.append(point1)

    top_down.add_line(point_list, colour=MAGENTA, width=1)

    return drawing_frame


def main_loop(vid: Video):
    if left_queue.empty() or right_queue.empty():
        time.sleep(0.01)
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

    raw_coords = localiser.get_coords(left_hand, right_hand)

    history.append(raw_coords)
    hand_coords = numpy.mean(history, axis=0)

    top_down.add_hand_points(hand_coords)

    frame_an = localiser.circle_3d_list(frame_an, hand_coords)

    ground_point = localiser.extract_point_of_interest(hand_coords)

    frame_an = draw_square_on_ground(frame_an, ground_point)

    top_down.add_line([hand_coords[8], ground_point], colour=MAGENTA, width=1)

    frame_an = localiser.line_3d(
        frame_an, [hand_coords[8], ground_point], colour=(0, 0, 255)
    )

    vid.show("Projection", frame_an)

    top_down_image = top_down.get_image()

    vid.show("Top Down", top_down_image)

    print(fps)


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
