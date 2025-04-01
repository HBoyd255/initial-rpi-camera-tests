import platform
import cv2
import numpy
from modules.eye import Eye

if platform.system() == "Windows":
    print("\nThis module only works on the RPi.\n")
    exit()


class Vision:

    def __init__(self):

        self.left = Eye("Left")
        self.right = Eye("Right")

    def both_arrays(self, lowres=True):
        return self.left.array(lowres=lowres), self.right.array(lowres=lowres)

    def joined_array(self, lowres=True):
        joined = numpy.hstack(
            (
                self.left.array(lowres=lowres),
                self.right.array(lowres=lowres),
            )
        )
        return joined

    def onion_array(self, lowres=True):
        onion = cv2.addWeighted(
            self.left.array(lowres=lowres),
            0.5,
            self.right.array(lowres=lowres),
            0.5,
            0,
        )

        return onion
