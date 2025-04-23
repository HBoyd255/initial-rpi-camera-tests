import numpy
from collections import deque
from modules.eye import Eye

from modules.localiser import Localiser
from modules.fps import FPS
from modules.video import Video


from typing import NamedTuple, cast
from modules.aruco import Aruco
from modules.zoomAruco import ZoomAruco

USE_THREADS = False

if USE_THREADS:
    from threading import Thread
    from queue import Queue
else:
    from multiprocessing import Process, Queue


class FrameStruct(NamedTuple):
    frame: numpy.ndarray
    aruco_list: list[Aruco]


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

        tags = tag_finder.get_tags(frame)

        frame = numpy.array(frame[::4, ::4])

        frame = tag_finder.draw_zoom_outline(frame)

        queue.put(FrameStruct(frame, tags))


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

        left_frame, left_tags = left_queue.get()
        right_frame, right_tags = right_queue.get()

        left_feed = numpy.copy(left_frame)
        right_feed = numpy.copy(right_frame)

        for tag in left_tags:
            left_feed = tag.draw(left_feed)

        for tag in right_tags:
            right_feed = tag.draw(right_feed)

        vid.show("Left Feed", left_feed)
        vid.show("Right Feed", right_feed)

        del left_feed, right_feed

        left_ids = {tag.id for tag in left_tags}
        right_ids = {tag.id for tag in right_tags}

        common_ids = left_ids & right_ids

        common_left_tags = [tag for tag in left_tags if tag.id in common_ids]
        common_right_tags = [tag for tag in right_tags if tag.id in common_ids]

        if not common_left_tags or not common_right_tags:
            print("Tags Missing")
            continue

        left_frame_an = numpy.copy(left_frame)

        if 19 in common_ids:
            left_tag = next(
                (tag for tag in common_left_tags if tag.id == 19), None
            )
            right_tag = next(
                (tag for tag in common_right_tags if tag.id == 19), None
            )
        else:
            left_tag = common_left_tags[0]
            right_tag = common_right_tags[0]

        tag_coords = localiser.get_coords(left_tag, right_tag)

        v1 = tag_coords[1] - tag_coords[0]
        v2 = tag_coords[3] - tag_coords[0]

        cross = numpy.cross(v1, v2)

        normal = cross / numpy.linalg.norm(cross)

        normal = -normal

        length = numpy.linalg.norm(v1)

        proj_points = tag_coords + (normal * length)

        left_frame_an = localiser.circle_3d(left_frame_an, proj_points[0])
        left_frame_an = localiser.circle_3d(left_frame_an, proj_points[1])
        left_frame_an = localiser.circle_3d(left_frame_an, proj_points[2])
        left_frame_an = localiser.circle_3d(left_frame_an, proj_points[3])

        left_frame_an = localiser.line_3d(left_frame_an, proj_points)
        left_frame_an = localiser.line_3d(left_frame_an, [proj_points[0],proj_points[3]])

        print()

        y_coords = tag_coords[:, 1]

        min_ = numpy.min(y_coords)
        max_ = numpy.max(y_coords)

        print(y_coords)
        print(max_ - min_)

        v1 = tag_coords[0] - tag_coords[3]
        v2 = tag_coords[1] - tag_coords[2]

        proj1 = numpy.copy(tag_coords[0])

        proj2 = numpy.copy(tag_coords[1])

        #         # else:
        #         for i in range(100):
        #             proj1 += v1
        #             proj2 += v2
        #
        #             if proj1[2] < 0:
        #                 break
        #
        #             left_frame_an = localiser.circle_3d(left_frame_an, proj1)
        #             left_frame_an = localiser.circle_3d(left_frame_an, proj2)
        #
        #         for p in tag_coords:
        #             left_frame_an = localiser.circle_3d(left_frame_an, p)

        vid.show("Projection", left_frame_an)


if __name__ == "__main__":

    if USE_THREADS:
        left_thread = Thread(target=capture_tag, args=("left", left_queue))
        right_thread = Thread(target=capture_tag, args=("right", right_queue))
        display_thread = Thread(target=show)

    # use process
    else:

        left_thread = Process(target=capture_tag, args=("left", left_queue))
        right_thread = Process(target=capture_tag, args=("right", right_queue))
        display_thread = Process(target=show)

    left_thread.start()
    right_thread.start()
    display_thread.start()

    left_thread.join()
    right_thread.join()
    display_thread.join()
