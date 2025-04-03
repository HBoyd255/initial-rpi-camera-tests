import os
import cv2


class Video:

    def __init__(self):

        self._display_available = "DISPLAY" in os.environ

    def show(self, name, frame):

        if not self._display_available:
            return

        cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF

        # If esc is pressed, exit the program
        if key == 27:
            exit()
