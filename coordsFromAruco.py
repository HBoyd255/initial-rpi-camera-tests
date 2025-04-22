import numpy
from collections import deque
from modules.eye import Eye
from multiprocessing import Process, Queue

from modules.localiser import Localiser
from modules.fps import FPS
from modules.video import Video
from modules.zoom import Zoom


from typing import NamedTuple, cast
from modules.aruco import Aruco
from modules.zoomAruco import ZoomAruco


class FrameStruct(NamedTuple):
    frame: numpy.ndarray
    tag: Aruco


left_queue = Queue(maxsize=1)
right_queue = Queue(maxsize=1)

# Cast for type checking and autocomplete
left_queue = cast("Queue[FrameStruct]", left_queue)
right_queue = cast("Queue[FrameStruct]", right_queue)

localiser = Localiser()


fps = FPS()


def capture_tag(side: str, queue: Queue):

    eye = Eye(side)

    tag_finder = ZoomAruco()

    while True:

        frame = eye.array(res="full")

        tag = tag_finder.get_tag(frame)

        frame = numpy.array(frame[::4, ::4])

        frame = tag_finder.draw_zoom_outline(frame)

        queue.put(FrameStruct(frame, tag))


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


history = deque(maxlen=10)


def show():

    vid = Video(canvas_framing=(2, 2))

    while True:

        if left_queue.empty() or right_queue.empty():
            continue

        left_frame, left_tag = left_queue.get()
        right_frame, right_tag = right_queue.get()

        left_feed = numpy.copy(left_frame)
        right_feed = numpy.copy(right_frame)

        left_feed = left_tag.draw(left_feed)
        right_feed = right_tag.draw(right_feed)

        vid.show("Left Feed", left_feed)
        vid.show("Right Feed", right_feed)

        del left_feed, right_feed

        left_frame_annotated = numpy.copy(left_frame)

        print(fps)

        if not (left_tag.is_seen() and right_tag.is_seen()):
            continue

        tag_coords = localiser.get_coords(left_tag, right_tag)

        print(tag_coords)

        for p in tag_coords:
            left_frame_annotated = localiser.circle_3d(left_frame_annotated, p)

        tip = tag_coords[1]

        knuckle = tag_coords[2]

        dif = tip - knuckle

        proj = numpy.copy(tip)

        for i in range(100):
            proj += dif

            if proj[2] < 0:
                break

            left_frame_annotated = localiser.circle_3d(
                left_frame_annotated, proj
            )

        ground_point = proj
        ground_point[2] = 0

        history.append(ground_point)

        point = numpy.average(history, axis=0)

        left_frame_annotated = draw_square_on_ground(
            left_frame_annotated, point
        )

        # left_frame_annotated = localiser.line_3d(
        #     left_frame_annotated, [tag_coords[8], point], colour=(0, 0, 255)
        # )

        vid.show("Projection", left_frame_annotated)


if __name__ == "__main__":

    left_thread = Process(target=capture_tag, args=("left", left_queue))
    right_thread = Process(target=capture_tag, args=("right", right_queue))

    display_thread = Process(target=show)

    left_thread.start()
    right_thread.start()
    display_thread.start()

    left_thread.join()
    right_thread.join()
    display_thread.join()
