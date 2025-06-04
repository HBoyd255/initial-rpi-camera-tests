import cv2
import numpy

from modules.colours import *


class TopDown:

    _CONE_POINTS = numpy.array(((-5, 4, 0), (0, 0, 0), (5, 4, 0)))
    _ROBOT_POINTS = numpy.array(
        (
            (-0.087, -0.28, 0),
            (0.183, -0.28, 0),
            (0.183, 0.05, 0),
            (-0.087, 0.05, 0),
            (-0.087, -0.28, 0),
        )
    )
    _90S_GRID = (
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

    def __init__(self, x_plot_bounds=(-0.5, 4), draw_grid=True):

        self._draw_grid = draw_grid

        # Sort the x range in forward order.
        self._x_plot_bounds = numpy.sort(x_plot_bounds)

        margin = 50

        self._frame_size = (576, 324)

        self._pixel_bounds = numpy.array(
            (
                [margin, self._frame_size[0] - margin],
                [margin, self._frame_size[1] - margin],
            )
        )

        frame_spans = numpy.array(
            (
                self._pixel_bounds[0][1] - self._pixel_bounds[0][0],
                self._pixel_bounds[1][1] - self._pixel_bounds[1][0],
            )
        )

        frame_aspect_ratio = frame_spans[0] / frame_spans[1]

        x_plot_span = numpy.max(self._x_plot_bounds) - numpy.min(
            self._x_plot_bounds
        )

        y_plot_span = x_plot_span / frame_aspect_ratio
        self._y_plot_bounds = numpy.sort((y_plot_span / 2, -y_plot_span / 2))

        plot_spans = numpy.array(
            (
                numpy.max(self._x_plot_bounds) - numpy.min(self._x_plot_bounds),
                numpy.max(self._y_plot_bounds) - numpy.min(self._y_plot_bounds),
            )
        )

        frame_aspect_ratio = frame_spans[0] / frame_spans[1]

        self.scale_coefficient = frame_spans / plot_spans
        self.plot_min = numpy.array(
            (numpy.min(self._x_plot_bounds), numpy.min(self._y_plot_bounds))
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

        sx = numpy.min(self._pixel_bounds[0])
        ex = numpy.max(self._pixel_bounds[0])

        sy = numpy.min(self._pixel_bounds[1])
        ey = numpy.max(self._pixel_bounds[1])

        p1 = numpy.array([sx, sy])
        p2 = numpy.array([ex, sy])
        p3 = numpy.array([ex, ey])
        p4 = numpy.array([sx, ey])

        points = numpy.array([p1, p2, p3, p4])

        cv2.circle(self._image, p1, 3, RED, -1)
        cv2.circle(self._image, p2, 3, RED, -1)
        cv2.circle(self._image, p3, 3, RED, -1)
        cv2.circle(self._image, p4, 3, RED, -1)

        TEMP = (self._y_plot_bounds[0], self._x_plot_bounds[0])
        self.add_labeled_point(TEMP)

        TEMP = (self._y_plot_bounds[1], self._x_plot_bounds[0])
        self.add_labeled_point(TEMP)

        TEMP = (self._y_plot_bounds[0], self._x_plot_bounds[1])
        self.add_labeled_point(TEMP)

        TEMP = (self._y_plot_bounds[1], self._x_plot_bounds[1])
        self.add_labeled_point(TEMP)

        cv2.polylines(self._image, [points], True, RED)

        for p in self._CONE_POINTS:
            self.add_point(p)

        for p in self._ROBOT_POINTS:
            self.add_point(p)

        if self._draw_grid:
            for p in self._90S_GRID:
                self.add_labeled_point(p)

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

    def add_labeled_point(self, point, colour=BLACK, offset=(-50, 20)):

        new_point = self._convert_coord(point)

        cv2.circle(self._image, new_point, 5, colour, -1)

        offset = numpy.array(offset, dtype=int)
        new_point += offset

        x_string = "{:.2f}".format(point[0])
        y_string = "{:.2f}".format(point[1])
        point_string = f"{x_string},{y_string}"

        if (len(point) > 2) and (point[2] != 0):
            z_string = "{:.2f}".format(point[2])
            point_string = f"{point_string},{z_string}"

        cv2.putText(
            img=self._image,
            text=point_string,
            org=new_point,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=0.5,
            color=colour,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    def add_point(self, point, colour=BLACK):

        new_point = self._convert_coord(point)

        cv2.circle(self._image, new_point, 5, colour, -1)

    def add_line(self, points, colour=BLACK, width=3):

        new_points = []

        for p in points:
            new_points.append(self._convert_coord(p))

        new_points = numpy.array(new_points, dtype=int)

        cv2.polylines(
            self._image,
            [new_points],
            isClosed=False,
            color=colour,
            thickness=width,
        )

    def add_hand_points(self, hand_coords):

        thumb = hand_coords[1:5]
        index = hand_coords[5:9]
        middle = hand_coords[9:13]
        ring = hand_coords[13:17]
        pinky = hand_coords[17:21]

        palm = hand_coords[[1, 0, 5, 9, 13, 17, 0]]

        self.add_line(palm, colour=PALM_COLOUR)

        self.add_line(pinky, colour=PINKY_COLOUR)
        self.add_line(ring, colour=RING_COLOUR)
        self.add_line(middle, colour=MIDDLE_COLOUR)
        self.add_line(index, colour=INDEX_COLOUR)
        self.add_line(thumb, colour=THUMB_COLOUR)

    def draw_radius(self, centre, radius):

        center_conv = self._convert_coord(centre)

        temp = (self.scale_coefficient * (radius, radius)).astype(int)

        cv2.ellipse(self._image, center_conv, temp, 0, 0, 360, BLACK)

        # print(center_conv)

    def get_image(self):

        image = numpy.copy(self._image)

        self._create()

        return image
