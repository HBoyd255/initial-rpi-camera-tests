import cv2
import numpy

from modules.colours import *


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
        y_plot_range = y_plot_range[::-1]

        # Sort the x range in forward order.
        x_plot_range = numpy.sort(x_plot_range)

        margin = 50

        self._frame_size = (576, 324)

        self._pixel_edges = numpy.array(
            (
                [margin, self._frame_size[0] - margin],
                [margin, self._frame_size[1] - margin],
            )
        )

        frame_range = numpy.array(
            (
                self._pixel_edges[0][1] - self._pixel_edges[0][0],
                self._pixel_edges[1][1] - self._pixel_edges[1][0],
            )
        )

        plot_range = numpy.array(
            (
                numpy.max(x_plot_range) - numpy.min(x_plot_range),
                numpy.max(y_plot_range) - numpy.min(y_plot_range),
            )
        )

        self.scale_coefficient = frame_range / plot_range
        self.plot_min = numpy.array(
            (numpy.min(x_plot_range), numpy.min(y_plot_range))
        )
        self.pixel_min = numpy.array((margin, margin))

        self._create()

    def _create(self):

        self._image = (
            numpy.ones(
                (self._frame_size[1], self._frame_size[0], 3), dtype=numpy.uint8
            )
            * 255
        )

        sx = numpy.min(self._pixel_edges[0])
        ex = numpy.max(self._pixel_edges[0])

        sy = numpy.min(self._pixel_edges[1])
        ey = numpy.max(self._pixel_edges[1])

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

        old_coord = numpy.array(old_coord)

        old_coord_reshaped = numpy.array(old_coord[[1, 0]])

        new_point = (
            ((old_coord_reshaped - self.plot_min) * self.scale_coefficient)
            + self.pixel_min
        ).astype(int)

        return new_point

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
        index = hand_coords[5:9]
        middle = hand_coords[9:13]
        ring = hand_coords[13:17]
        pinky = hand_coords[17:21]

        palm = hand_coords[[1, 0, 5, 9, 13, 17, 0]]

        self.add_line(thumb, colour=THUMB_COLOUR)
        self.add_line(index, colour=INDEX_COLOUR)
        self.add_line(middle, colour=MIDDLE_COLOUR)
        self.add_line(ring, colour=RING_COLOUR)
        self.add_line(pinky, colour=PINKY_COLOUR)

        self.add_line(palm, colour=PALM_COLOUR)

    def get_image(self):

        image = numpy.copy(self._image)

        self._create()

        return image
