import numpy
from modules.colours import *
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

top_down = TopDown(
    x_range=(-0.2, 0.2),
    y_range=(0.30, 0.7),
)


def capture_hand(side: str, queue: Queue):

    eye = Eye(side)

    hand_finder = Zoom()
    while True:

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

    drawing_frame = localiser.line_3d(
        drawing_frame,
        [point1, point2, point3, point4],
    )

    return drawing_frame


def show():

    vid = Video(canvas_framing=(2, 2))

    while True:

        if left_queue.empty() or right_queue.empty():
            continue

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

        print(fps)

        if not (left_hand.is_seen() and right_hand.is_seen()):
            continue

        hand_coords = localiser.get_coords(left_hand, right_hand)

        top_down.add_hand_points(hand_coords)

        for p in hand_coords:
            frame_an = localiser.circle_3d(frame_an, p)
            top_down.add_point(p)

        centre = (hand_coords[0] + hand_coords[9]) / 2

        v1 = hand_coords[5] - hand_coords[0]
        v2 = hand_coords[17] - hand_coords[0]

        cross = numpy.cross(v1, v2)

        normal = cross / numpy.linalg.norm(cross)

        tip = centre + (normal * 0.1)

        frame_an = localiser.circle_3d(frame_an, centre, colour=GREEN_BGR)
        top_down.add_point(centre, colour=GREEN_NORMAL)

        frame_an = localiser.circle_3d(frame_an, tip, colour=MAGENTA_BGR)
        top_down.add_point(tip, colour=MAGENTA_NORMAL)

        top_down.add_point(p)

        vid.show("Projection", frame_an)

        top_down_image = top_down.get_image()

        vid.show("Top Down", top_down_image)


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
