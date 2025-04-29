from collections import deque
import time
import cv2
import numpy
from modules.colours import *
from modules.evaluateVariable import evaluate_variable
from modules.eye import Eye

from modules.gesture import GestureClassifier
from modules.localiser import Localiser
from modules.fps import FPS
from modules.topDown import TopDown
from modules.video import Video
from modules.zoom import Zoom
from modules.wheels import Wheels


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
top_full = TopDown(x_plot_bounds=(-0.5, 4.5))


history = deque(maxlen=10)

wheel_control = Wheels()


def normalize_speeds(speeds: numpy.ndarray) -> numpy.ndarray:

    # Find the maximum absolute value
    max_value = numpy.max(numpy.abs(speeds))

    if max_value > 100:
        speeds = speeds / max_value * 100

        speeds = speeds.astype(int)

    return speeds


def probabilistic(power):

    sine = -1
    if power > 0:
        sine = 1

    trip_power = 80

    rand = numpy.random.randint(0, trip_power)

    if abs(power) > rand:
        return trip_power * sine

    return 0


def match_distance_and_angle(hand: Hand, hand_points):
    wrist_distance = hand_points[0][1]

    rx = hand.landmarks[0][0]

    vertical_error = (wrist_distance - 1) * 2
    horizontal_error = (0.5 - rx) * 5

    vertical_error = int(vertical_error * 100)
    horizontal_error = int(horizontal_error * 100)

    vertical_error = probabilistic(vertical_error)
    horizontal_error = probabilistic(horizontal_error)

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


def match_angle(hand: Hand):

    rx = hand.landmarks[0][0]

    horizontal_error = (0.5 - rx) * 5

    horizontal_error = int(horizontal_error * 100)

    horizontal_error = probabilistic(horizontal_error)

    speeds = numpy.zeros(4, dtype=int)

    speeds += (
        -horizontal_error,
        horizontal_error,
        -horizontal_error,
        horizontal_error,
    )

    speeds = normalize_speeds(speeds)

    wheel_control.send(*speeds)


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
    top_full.add_line(point_list, colour=MAGENTA, width=1)

    return drawing_frame


gest = GestureClassifier()


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

    raw_coords = localiser.get_coords(left_hand, right_hand)

    history.append(raw_coords)
    hand_coords = numpy.mean(history, axis=0)

    frame_an = localiser.circle_3d_list(frame_an, raw_coords, colour=RED)
    frame_an = localiser.circle_3d_list(frame_an, hand_coords, colour=BLUE)

    gesture_name = gest.get_gesture_name(raw_coords)

    print(gesture_name)

    user_is_pointing = gesture_name == "Index"
    double_point = gesture_name == "German Three"
    # flipped_off = gesture_name == "Middle"
    # is_spider_man = gesture_name == "Spider-Man"
    user_holding_fist = gesture_name == "Fist"
    puppy_dog_mode = gesture_name == "Rock On"

    top_down.add_hand_points(hand_coords)
    top_full.add_hand_points(hand_coords)

    if user_is_pointing or double_point:

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

        if dotted_line:
            frame_an = localiser.circle_3d_list(frame_an, dotted_line)

        frame_an = draw_square_on_ground(frame_an, ground_point)

        top_down.add_line(
            [hand_coords[8], ground_point], colour=MAGENTA, width=1
        )
        top_full.add_line(
            [hand_coords[8], ground_point], colour=MAGENTA, width=1
        )

        frame_an = localiser.line_3d(
            frame_an, [hand_coords[8], ground_point], colour=(0, 0, 255)
        )

        if double_point:

            az_poi = numpy.arctan2(ground_point[0], ground_point[1])

            angle_error = -numpy.rad2deg(az_poi) * 3

            angle_error = probabilistic(angle_error)

            speeds = numpy.zeros(4, dtype=float)

            speeds += (
                -angle_error,
                angle_error,
                -angle_error,
                angle_error,
            )

            speeds = speeds.astype(int)

            print(speeds)

            wheel_control.send(*speeds)

            print()

    #     if flipped_off:
    #         print("Thats rude, im leaving")
    #         wheel_control.send(-80, -80, -80, -80)
    #
    #     if is_spider_man:
    #         print("I love Spider-Man")
    #         wheel_control.send(80, 80, 80, 80)

    if user_holding_fist:
        match_angle(left_hand)

    if puppy_dog_mode:
        match_distance_and_angle(left_hand, raw_coords)

    vid.show("Projection", frame_an)

    top_down_image = top_down.get_image()

    vid.show("Top Down", top_down_image)

    
    top_down_image_full = top_full.get_image()

    vid.show("Top Down Full", top_down_image_full)

    print(fps)


def show():

    vid = Video(canvas_framing=(3,2))

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
