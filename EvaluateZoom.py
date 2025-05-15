from functools import lru_cache
import cv2

import numpy
from modules.duration import Duration
from modules.eye import Eye
from modules.video import Video
from modules.zoomLive import Zoom

vid = Video(canvas_framing=(1, 1))
eye = Eye("left")

hand_finder = Zoom(continuous=True)


hands_list = []


def get_frame(i):
    path = f"data/video/frame_{i}.png"

    frame = cv2.imread(path)

    return frame


dura = Duration()

dura.head()

print("Time to load images from memory and convert to RGB again")
for i in range(100):

    print(f"{i}/100")

    frame = get_frame(i)

    frame_small = numpy.array(frame[::4, ::4], dtype=numpy.uint8)

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

dura.flag()
print("Time to load images from memory and convert to RGB again")
for i in range(100):

    print(f"{i}/100")

    frame = get_frame(i)

    frame_small = numpy.array(frame[::4, ::4], dtype=numpy.uint8)

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

dura.flag()

print("Time to load images from memory, and prossess.")
for i in range(100):

    print(f"{i}/100")

    frame = get_frame(i)

    hand = hand_finder.get_hand(frame)

    hands_list.append(hand)

# dura.flag()
# 
# print("Time to load images from cashe, draw results and save.")
# for i in range(100):
# 
#     print(f"{i}/100")
# 
#     frame = get_frame(i)
# 
#     hand = hands_list[i]
# 
#     frame = hand.draw(frame)
# 
#     path = f"data/video_Zoom/frame_{i}.png"
# 
#     cv2.imwrite(path, frame)


dura.flag()
dura.head()


successes = [hand.is_seen() for hand in hands_list]

count = numpy.sum(successes)

print(successes)
print(count)

# Static mode
# 75%

# EvaluateZoom.py:29 -> EvaluateZoom.py:42 = 7433.44ms.
# EvaluateZoom.py:42 -> EvaluateZoom.py:54 = 7246.88ms.
# EvaluateZoom.py:54 -> EvaluateZoom.py:85 = 16596.01ms.


# 9349.13 for 100
# 93.49 for 1


# Continuous Mode
# 76% 

# EvaluateZoom.py:29 -> EvaluateZoom.py:42 = 7166.38ms.
# EvaluateZoom.py:42 -> EvaluateZoom.py:54 = 7121.85ms.
# EvaluateZoom.py:54 -> EvaluateZoom.py:85 = 12912.03ms.

# 5790.18 for 100
# 57.9018 for 1

