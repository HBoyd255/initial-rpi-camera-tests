from collections import namedtuple
import numpy
from modules.eye import Eye
from multiprocessing import Process, Queue

from modules.localiser import Localiser
from modules.fps import FPS
from modules.video import Video
from modules.zoom import Zoom

FrameStruct = namedtuple("FrameStruct", ["frame", "hand"])

left_queue = Queue(maxsize=1)
right_queue = Queue(maxsize=1)

localiser = Localiser()


fps = FPS()


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

        print(fps)

        if not (left_hand.is_seen() and right_hand.is_seen()):
            continue

        hand_coords = localiser.get_coords(left_hand, right_hand)

        print(hand_coords[8])

        for i in range(21):

            p = hand_coords[i]

            left_feed = localiser.circle_3d(left_feed, p)

        tip = hand_coords[8]

        knuckle = hand_coords[5]

        dif = tip - knuckle

        proj = numpy.copy(tip)

        if dif[2] < 0:

            while True:
                proj += dif

                if proj[2] < 0:
                    break

                left_feed = localiser.circle_3d(left_feed, proj)

            ground_point = proj
            ground_point[2] = 0

            left_feed = draw_square_on_ground(left_feed, ground_point)

            left_feed = localiser.line_3d(
                left_feed, [hand_coords[8], ground_point], colour=(0, 0, 255)
            )

        vid.show("Projection", left_feed)


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
