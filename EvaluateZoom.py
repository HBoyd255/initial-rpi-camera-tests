from functools import lru_cache
import cv2

import numpy
from modules.duration import Duration
from modules.eye import Eye
from modules.video import Video
from modules.zoomLive import Zoom

vid = Video(canvas_framing=(1, 1))
eye = Eye("left")

hand_finder = Zoom(continuous=False)


hands_list = []


def get_frame(i):
    path = f"data/video/frame_{i}.png"

    frame = cv2.imread(path)

    return frame


dura = Duration()

dura.head()


print("Time to load images from memory and convert to RGB")
for i in range(100):

    print(f"{i}/100")

    frame = get_frame(i)
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

dura.flag()

print("Time to load images from memory and convert to RGB again")
for i in range(100):

    print(f"{i}/100")

    frame = get_frame(i)
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


# continuous=False


# EvaluateZoom.py:40 -> EvaluateZoom.py:50 = 6941.76ms.
# EvaluateZoom.py:50 -> EvaluateZoom.py:81 = 15154.96ms.


# 15154.96ms - 6941.76ms = 8213.2ms

# 8213.2 ms for for 100 images

# 82.132 ms per frame

# 12.1755223299 FPS
# with 96% success


# continuous=True

# EvaluateZoom.py:42 -> EvaluateZoom.py:52 = 7004.83ms.
# EvaluateZoom.py:52 -> EvaluateZoom.py:65 = 10373.51ms.


# 10373.51ms. - 7004.83ms. = 3368.68

# 3368.68 ms for 100 images
# 33.6868 ms per frame.

# 29.6852179489
# with 96% success
