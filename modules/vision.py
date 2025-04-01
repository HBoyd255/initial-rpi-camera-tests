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

    def both_arrays(self, res="low"):
        return self.left.array(res=res), self.right.array(res=res)

    def joined_array(self, res="low"):
        joined = numpy.hstack(
            (
                self.left.array(res=res),
                self.right.array(res=res),
            )
        )
        return joined

    def onion_array(self, res="low"):
        onion = cv2.addWeighted(
            self.left.array(res=res),
            0.5,
            self.right.array(res=res),
            0.5,
            0,
        )

        return onion

    def anaglyph_array(self, res="low"):

        left_array, right_array = self.both_arrays(res=res)

        anaglyph = numpy.zeros_like(left_array)

        anaglyph[..., 2] = left_array[..., 2]
        # anaglyph[..., 1] = right_array[..., 1]
        anaglyph[..., 0] = right_array[..., 0]

        return anaglyph
