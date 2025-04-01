import cv2
import numpy
from modules.vision import Vision


eyes = Vision()


while True:

    anaglyph = eyes.anaglyph_array(res="mid")

    frame_height, frame_width, _ = anaglyph.shape

    screen_width = 1920
    screen_height = 1080

    x_off = (screen_width - frame_width) // 2
    y_off = (screen_height - frame_height) // 2

    screen = numpy.zeros((screen_height, screen_width, 3), dtype=numpy.uint8)
    screen[y_off : y_off + frame_height, x_off : x_off + frame_width] = anaglyph

    cv2.imshow("anaglyph", screen)
    cv2.namedWindow("anaglyph", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "anaglyph", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    cv2.waitKey(1)
