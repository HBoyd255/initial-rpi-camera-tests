import numpy


def normalise(rgb):
    return tuple(float(c) for c in numpy.array(rgb) / 255)


def to_bgr(rgb):
    return tuple(int(c) for c in numpy.array(rgb)[::-1])


RED = (255, 0, 0)
RED_NORMAL = normalise(RED)
RED_BGR = to_bgr(RED)

GREEN = (0, 255, 0)
GREEN_NORMAL = normalise(GREEN)
GREEN_BGR = to_bgr(GREEN)

BLUE = (0, 0, 255)
BLUE_NORMAL = normalise(BLUE)
BLUE_BGR = to_bgr(BLUE)

MAGENTA = (255, 0, 255)
MAGENTA_NORMAL = normalise(MAGENTA)
MAGENTA_BGR = to_bgr(MAGENTA)

WHITE = (255, 255, 255)
WHITE_NORMAL = normalise(WHITE)
WHITE_BGR = to_bgr(WHITE)

BLACK = (0, 0, 0)
BLACK_NORMAL = normalise(BLACK)
BLACK_BGR = to_bgr(BLACK)


# ----------------------------

THUMB_COLOUR = BLUE
THUMB_COLOUR_NORMAL = normalise(THUMB_COLOUR)
THUMB_COLOUR_BGR = to_bgr(THUMB_COLOUR)


INDEX_COLOUR = RED
INDEX_COLOUR_NORMAL = normalise(INDEX_COLOUR)
INDEX_COLOUR_BGR = to_bgr(INDEX_COLOUR)

MIDDLE_COLOUR = (108, 60, 200)
MIDDLE_COLOUR_NORMAL = normalise(MIDDLE_COLOUR)
MIDDLE_COLOUR_BGR = to_bgr(MIDDLE_COLOUR)

RING_COLOUR = GREEN
RING_COLOUR_NORMAL = normalise(RING_COLOUR)
RING_COLOUR_BGR = to_bgr(RING_COLOUR)

PINKY_COLOUR = (255, 32, 193)
PINKY_COLOUR_NORMAL = normalise(PINKY_COLOUR)
PINKY_COLOUR_BGR = to_bgr(PINKY_COLOUR)

PALM_COLOUR = (123, 123, 125)
PALM_COLOUR_NORMAL = normalise(PALM_COLOUR)
PALM_COLOUR_BGR = to_bgr(PALM_COLOUR)
