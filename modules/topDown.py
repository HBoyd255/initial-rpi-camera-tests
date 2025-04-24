import cv2
import numpy

from modules.colours import *

# from video import Video


class TopDown:

    _CONE_POINTS = numpy.array([[-5, 4, 0], [0, 0, 0], [5, 4, 0]])
    _ROBOT_POINTS = numpy.array(
        [
            [-0.135, -0.28, 0],
            [0.135, -0.28, 0],
            [0.135, 0.05, 0],
            [-0.135, 0.05, 0],
            [-0.135, -0.28, 0],
        ]
    )

    def __init__(self, y_plot_range=(2, -2), x_plot_range=(-0.5, 4)):

        # Sort the x range in reverse order.
        y_plot_range = numpy.sort(y_plot_range)
        self._y_plot_range = y_plot_range[::-1]

        # Sort the x range in forward order.
        self._x_plot_range = numpy.sort(x_plot_range)

        self._frame_width = 576
        self._frame_height = 324

        self._y_pixel_edges = [50, self._frame_height - 50]
        self._x_pixel_edges = [50, self._frame_width - 50]

        self._create()

    def _create(self):

        self._image = (
            numpy.ones(
                (self._frame_height, self._frame_width, 3), dtype=numpy.uint8
            )
            * 255
        )

        sx = numpy.min(self._x_pixel_edges)
        ex = numpy.max(self._x_pixel_edges)

        sy = numpy.min(self._y_pixel_edges)
        ey = numpy.max(self._y_pixel_edges)

        p1 = numpy.array([sx, sy])
        p2 = numpy.array([ex, sy])
        p3 = numpy.array([ex, ey])
        p4 = numpy.array([sx, ey])

        points = numpy.array([p1, p2, p3, p4])

        cv2.circle(self._image, p1, 3, RED, -1)
        cv2.circle(self._image, p2, 3, RED, -1)
        cv2.circle(self._image, p3, 3, RED, -1)
        cv2.circle(self._image, p4, 3, RED, -1)

        cv2.polylines(self._image, [points], True, RED)

        for p in self._CONE_POINTS:
            self.add_point(p)

        for p in self._ROBOT_POINTS:
            self.add_point(p)

        self.add_line(self._CONE_POINTS)
        self.add_line(self._ROBOT_POINTS)

    def _convert_coord(self, old_coord):

        old_y = old_coord[0]
        old_x = old_coord[1]

        plot_ry = numpy.max(self._y_plot_range) - numpy.min(self._y_plot_range)
        frame_ry = numpy.max(self._y_pixel_edges) - numpy.min(
            self._y_pixel_edges
        )

        plot_min_y = numpy.min(self._y_plot_range)
        pixel_min_y = numpy.min(self._y_pixel_edges)

        new_y = int(((old_y - plot_min_y) * frame_ry / plot_ry) + pixel_min_y)

        plot_rx = numpy.max(self._x_plot_range) - numpy.min(self._x_plot_range)
        frame_rx = numpy.max(self._x_pixel_edges) - numpy.min(
            self._x_pixel_edges
        )

        plot_min_x = numpy.min(self._x_plot_range)
        pixel_min_x = numpy.min(self._x_pixel_edges)

        new_x = int(((old_x - plot_min_x) * frame_rx / plot_rx) + pixel_min_x)

        return [new_x, new_y]

    def add_point(self, point, colour=BLACK):

        new_point = self._convert_coord(point)

        cv2.circle(self._image, new_point, 5, colour, -1)

        # self._ax.plot(point[1], , color=colour, marker="o")

    def add_line(self, points, colour=BLACK):

        new_points = []

        for p in points:
            new_points.append(self._convert_coord(p))

        new_points = numpy.array(new_points, dtype=int)

        # print(new_points)

        cv2.polylines(
            self._image, [new_points], isClosed=False, color=colour, thickness=3
        )

    def add_hand_points(self, hand_coords):

        thumb = hand_coords[1:5]
        self.add_line(thumb, colour=THUMB_COLOUR)

        index = hand_coords[5:9]
        self.add_line(index, colour=INDEX_COLOUR)

        middle = hand_coords[9:13]
        self.add_line(middle, colour=MIDDLE_COLOUR)

        ring = hand_coords[13:17]
        self.add_line(ring, colour=RING_COLOUR)

        pinky = hand_coords[17:21]

        self.add_line(pinky, colour=PINKY_COLOUR)

        palm = hand_coords[[1, 0, 5, 9, 13, 17, 0]]
        self.add_line(palm, colour=PALM_COLOUR)

    def get_image(self):

        image = numpy.copy(self._image)

        self._create()

        return image
