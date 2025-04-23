import cv2
import numpy
import matplotlib.pyplot as plt

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

    def __init__(self, x_range=(2, -2), y_range=(-0.5, 4)):

        # Sort the x range in reverse order.
        x_range = numpy.sort(x_range)
        x_range = x_range[::-1]

        # Sort the x range in forward order.
        y_range = numpy.sort(y_range)

        self._xlim = y_range
        self._ylim = x_range

        self._create()

    def _create(self):

        fig, ax = plt.subplots()

        self._fig = fig
        self._ax = ax

        ax.set_xlim(self._xlim)
        ax.set_ylim(self._ylim)
        ax.set_aspect("equal")

        self.add_line(self._CONE_POINTS, BLACK_NORMAL)
        self.add_line(self._ROBOT_POINTS, BLACK_NORMAL)

    def add_point(self, point, colour=BLACK_NORMAL):

        self._ax.plot(point[1], point[0], color=colour, marker="o")

    def add_line(self, points, colour=BLACK_NORMAL):

        self._ax.plot(points[:, 1], points[:, 0], color=colour)

    def add_hand_points(self, hand_coords):

        thumb = hand_coords[1:5]
        self.add_line(thumb, colour=THUMB_COLOUR_NORMAL)

        index = hand_coords[5:9]
        self.add_line(index, colour=INDEX_COLOUR_NORMAL)

        middle = hand_coords[9:13]
        self.add_line(middle, colour=MIDDLE_COLOUR_NORMAL)

        ring = hand_coords[13:17]
        self.add_line(ring, colour=RING_COLOUR_NORMAL)

        pinky = hand_coords[17:21]

        self.add_line(pinky, colour=PINKY_COLOUR_NORMAL)

        palm = hand_coords[[1, 0, 5, 9, 13, 17, 0]]
        self.add_line(palm, colour=PALM_COLOUR_NORMAL)

    def get_image(self):

        self._fig.canvas.draw()

        image = numpy.array(self._fig.canvas.renderer.buffer_rgba())

        image = cv2.resize(image, (576, 324))

        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        self._destroy()
        self._create()

        return image

    def _destroy(self):
        plt.close(self._fig)
