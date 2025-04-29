from collections import deque
import time
import cv2
import numpy
from modules.colours import *
from modules.evaluateVariable import evaluate_variable
from modules.eye import Eye

from modules.localiser import Localiser
from modules.fps import FPS
from modules.duration import Duration
from modules.physical import *
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

# top_down = TopDown(
#     y_plot_range=(-0.25, 0.25),
#     x_plot_range=(-0.25, 0.75),
# )

# top_down = TopDown(
#     y_plot_range=(0.5, -1.5),
#     x_plot_range=(-0.5, 4.5),
# )

top_down = TopDown(
    y_plot_range=(0.5, -1.5),
    x_plot_range=(-0.5, 4.5),
)

history = deque(maxlen=10)


def capture_hand(side: str, queue: Queue):

    eye = Eye(side)

    hand_finder = Zoom(continuous=True)
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

    top_down.add_line(point_list, colour=MAGENTA, width=1)

    return drawing_frame


dura = Duration(kill=True)


def main_loop(vid: Video):

    if left_queue.empty() or right_queue.empty():
        return

    dura.head()

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

    # history.append(raw_coords)
    # hand_coords = numpy.mean(history, axis=0)

    hand_coords = numpy.copy(raw_coords)

    top_down.add_hand_points(hand_coords)

    frame_an = localiser.circle_3d_list(frame_an, hand_coords)

    # --------------------------------
    wrist_3d = hand_coords[0]
    writst_from_camera = wrist_3d - (0, 0, 0.192)

    wrist_r = numpy.linalg.norm(writst_from_camera)

    top_down.draw_radius((0, 0), wrist_r)

    wrist_azm = numpy.arctan2(writst_from_camera[0], writst_from_camera[1])
    wrist_elv = numpy.arcsin(writst_from_camera[2] / wrist_r)
    wrist_projected = numpy.array((0.0, 0.0, 0.0))

    wrist_projected[0] = wrist_r * numpy.cos(wrist_elv) * numpy.sin(wrist_azm)
    wrist_projected[1] = wrist_r * numpy.cos(wrist_elv) * numpy.cos(wrist_azm)
    wrist_projected[2] = (wrist_r * numpy.sin(wrist_elv)) + 0.192

    # print(wrist_3d)
    # print(wrist_projected)
    # print(wrist_projected - wrist_3d)
    # print()

    # print(wrist_projected)

    top_down.add_point(wrist_projected, colour=RED)
    frame_an = localiser.circle_3d(frame_an, wrist_projected, colour=RED)

    tip_3d = hand_coords[8]
    tip_from_camera = tip_3d - (0, 0, 0.192)

    tip_r = numpy.linalg.norm(tip_from_camera)

    top_down.draw_radius((0, 0), wrist_r)

    tip_azm = numpy.arctan2(tip_from_camera[0], tip_from_camera[1])
    tip_elv = numpy.arcsin(tip_from_camera[2] / tip_r)

    tip_projected = numpy.array((0.0, 0.0, 0.0))

    tip_projected[0] = wrist_r * numpy.cos(tip_elv) * numpy.sin(tip_azm)
    tip_projected[1] = wrist_r * numpy.cos(tip_elv) * numpy.cos(tip_azm)
    tip_projected[2] = (wrist_r * numpy.sin(tip_elv)) + 0.192

    top_down.add_point(tip_projected, colour=RED)
    frame_an = localiser.circle_3d(frame_an, tip_projected, colour=RED)

    projected_finger_v = tip_projected - wrist_projected

    projected_finger_length = numpy.linalg.norm(projected_finger_v)

    # print(projected_finger_length)

    finger_azm = numpy.arctan2(projected_finger_v[0], projected_finger_v[1])
    finger_elv = numpy.arcsin(projected_finger_v[2] / projected_finger_length)

    # top_down.add_line(((0, 0, 0), proj_wrist_point), width=1)

    #         tip_ang_h = (tip_2d[0] - 0.5) * HORIZONTAL_FOV_DEGREES
    #
    #         proj_tip_point = [0, 0, 0]
    #         proj_tip_point[0] = top_down_r * numpy.sin(numpy.radians(tip_ang_h))
    #         proj_tip_point[1] = top_down_r * numpy.cos(numpy.radians(tip_ang_h))
    #         top_down.add_point(proj_tip_point)
    #         top_down.add_line(((0, 0, 0), proj_tip_point), width=1)
    #
    #         top_down.draw_radius(proj_wrist_point, 0.17)
    #
    #         dura.flag()

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

    frame_an = localiser.circle_3d_list(frame_an, points_to_add)

    for p in points_to_add:

        top_down.add_point(p)

    cv2.putText(
        frame_an,
        f"Finger = {projected_finger_length:.4}",
        (30, 100),
        cv2.FONT_HERSHEY_TRIPLEX,
        1.5,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        frame_an,
        f"Angle = {numpy.rad2deg(finger_azm):.4}",
        (30, 200),
        cv2.FONT_HERSHEY_TRIPLEX,
        1.5,
        (0, 0, 255),
        2,
    )

    dotted_line = []

    proj = numpy.copy(tip_projected)
    dif = numpy.copy(projected_finger_v)

    for i in range(100):
        proj += dif

        if proj[2] < 0:
            break

        dotted_line.append(numpy.copy(proj))

    ground_point = proj
    ground_point[2] = 0

    if dotted_line:
        frame_an = localiser.circle_3d_list(frame_an, dotted_line)

    top_down.add_line([tip_projected, ground_point], colour=MAGENTA, width=1)

    frame_an = localiser.line_3d(
        frame_an, [tip_projected, ground_point], colour=(0, 0, 255)
    )

    dura.flag()
    frame_an = draw_square_on_ground(frame_an, ground_point)

    vid.show("Projection", frame_an)

    top_down_image = top_down.get_image()

    vid.show("Top Down", top_down_image)

    dura.flag()

    # print(fps)


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
