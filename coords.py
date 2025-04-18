from collections import namedtuple
import numpy
from modules.eye import Eye
from multiprocessing import Process, Queue

from modules.localiser import Localiser
from modules.fps import FPS
from modules.video import Video
from modules.zoom import Zoom, draw_zoom_outline

FrameStruct = namedtuple("FrameStruct", ["frame", "hand"])

left_queue = Queue(maxsize=1)
right_queue = Queue(maxsize=1)

BASELINE_M = 0.096
FOCAL_LENGTH_PX = 1964
FRAME_WIDTH_PX = 4608

HORIZONTAL_FOV_DEGREES = 102
VERTICAL_FOV_DEGREES = 67

ELEVATION_OFFSET_DEGREES = -21
HORIZONTAL_OFFSET = 0.192


localiser = Localiser(
    BASELINE_M,
    FOCAL_LENGTH_PX,
    FRAME_WIDTH_PX,
    HORIZONTAL_FOV_DEGREES,
    VERTICAL_FOV_DEGREES,
    ELEVATION_OFFSET_DEGREES,
    HORIZONTAL_OFFSET,
)


fps = FPS()


def capture_hand(side: str, queue: Queue):

    eye = Eye(side)

    hand_finder = Zoom()
    while True:

        frame = eye.array(res="full")

        hand = hand_finder.get_hand(frame, simple=False)

        frame = numpy.array(frame[::4, ::4])

        draw_zoom_outline(frame, hand_finder._zoom_coords)

        queue.put(FrameStruct(frame, hand))


def draw_square_on_ground(frame, ground_coord):

    drawing_frame = numpy.copy(frame)

    ground_coord = numpy.array(ground_coord)

    point1 = ground_coord + [0.1, 0.1, 0]
    point2 = ground_coord + [-0.1, 0.1, 0]
    point3 = ground_coord + [-0.1, -0.1, 0]
    point4 = ground_coord + [0.1, -0.1, 0]

    drawing_frame = localiser.line_3d(drawing_frame, point1, point2)
    drawing_frame = localiser.line_3d(drawing_frame, point2, point3)
    drawing_frame = localiser.line_3d(drawing_frame, point3, point4)
    drawing_frame = localiser.line_3d(drawing_frame, point4, point1)

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

        if not (left_hand.is_seen() and right_hand.is_seen()):
            continue

        hand_coords = localiser.get_coords(left_hand, right_hand)

        for i in range(11):
            line = numpy.array([[-1, i * 0.2, 0], [1, i * 0.2, 0]])
            left_feed = localiser.line_3d(left_feed, line[0], line[1])

            line = numpy.array([[(i * 0.2) - 1, 1, 0], [(i * 0.2) - 1, 2, 0]])
            left_feed = localiser.line_3d(left_feed, line[0], line[1])

        for i in range(21):

            p = hand_coords[i]

            left_feed = localiser.circle_3d(left_feed, p)

        tip = hand_coords[8]

        knuckle = hand_coords[5]

        dif = tip - knuckle

        dif /= 3

        proj = tip

        if dif[2] < 0:

            while True:
                proj += dif

                if proj[2] < 0:
                    break

                left_feed = localiser.circle_3d(left_feed, proj)

            ground_point = proj
            ground_point[2] = 0

            # left_feed = line_3d(left_feed, coords[8], peebs, colour=(0, 0, 255))

            left_feed = draw_square_on_ground(left_feed, ground_point)

        print(fps)
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
