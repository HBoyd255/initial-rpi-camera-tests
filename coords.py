import time
import numpy
from modules.colours import *
from modules.evaluateVariable import evaluate_variable
from modules.eye import Eye

from modules.localiser import Localiser
from modules.fps import FPS
from modules.duration import Duration
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
    y_plot_range=(0.5, -1.5),
    x_plot_range=(-0.5, 4.5),
)


dura = Duration(kill=True)


def capture_hand(side: str, queue: Queue):

    eye = Eye(side)

    hand_finder = Zoom()
    while True:

        if queue.full():
            time.sleep(0.1)

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

    top_down.add_line(point_list, colour=MAGENTA)

    return drawing_frame


def main_loop(vid):
    if left_queue.empty() or right_queue.empty():
        return

    dura.head()

    left_frame, left_hand = left_queue.get()
    right_frame, right_hand = right_queue.get()

    left_feed = numpy.copy(left_frame)
    right_feed = numpy.copy(right_frame)

    left_feed = left_hand.draw(left_feed)
    right_feed = right_hand.draw(right_feed)

    dura.flag()

    vid.show("Left Feed", left_feed)
    vid.show("Right Feed", right_feed)

    dura.flag()

    del left_feed, right_feed

    frame_an = numpy.copy(left_frame)

    if not (left_hand.is_seen() and right_hand.is_seen()):
        return

    dura.flag()

    hand_coords = localiser.get_coords(left_hand, right_hand)

    dura.flag()

    top_down.add_hand_points(hand_coords)

    dura.flag()

    frame_an = localiser.circle_3d_list(frame_an, hand_coords)

    dura.flag()

    for p in hand_coords:
        top_down.add_point(p)

    dura.flag()

    centre = (hand_coords[0] + hand_coords[9]) / 2

    v1 = hand_coords[5] - hand_coords[0]
    v2 = hand_coords[17] - hand_coords[0]

    cross = numpy.cross(v1, v2)

    normal = cross / numpy.linalg.norm(cross)

    tip = centre + (normal * 0.1)

    frame_an = localiser.circle_3d(frame_an, centre, colour=GREEN)
    top_down.add_point(centre, colour=GREEN)

    frame_an = localiser.circle_3d(frame_an, tip, colour=MAGENTA)
    top_down.add_point(tip, colour=MAGENTA)

    dura.flag()

    tip = hand_coords[8]

    knuckle = hand_coords[5]

    dif = tip - knuckle

    proj = numpy.copy(tip)

    dotted_line = []

    for i in range(100):
        proj += dif

        if proj[2] < 0:
            break

        dotted_line.append(numpy.copy(proj))

    ground_point = proj
    ground_point[2] = 0

    dura.flag()

    if dotted_line:
        frame_an = localiser.circle_3d_list(frame_an, dotted_line)

    dura.flag()
    frame_an = draw_square_on_ground(frame_an, ground_point)

    dura.flag()

    top_down.add_line([hand_coords[8], ground_point], colour=MAGENTA)

    frame_an = localiser.line_3d(
        frame_an, [hand_coords[8], ground_point], colour=(0, 0, 255)
    )

    dura.flag()

    vid.show("Projection", frame_an)

    points_to_add = (
        (0, 0, 0),
        (0, 0.9, 0),
        (0, 1.8, 0),
        (0, 2.7, 0),
        (0, 3.6, 0),
        (-0.9, 0, 0),
        (-0.9, 0.9, 0),
        (-0.9, 1.8, 0),
        (-0.9, 2.7, 0),
        (-0.9, 3.6, 0),
    )

    for p in points_to_add:

        top_down.add_point(p)

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
