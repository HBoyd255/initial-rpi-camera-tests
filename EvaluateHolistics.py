from functools import lru_cache
import time
import cv2

import mediapipe
import numpy
from modules.duration import Duration
from modules.eye import Eye
from modules.video import Video

vid = Video(canvas_framing=(1, 1))
eye = Eye("left")


mp_holistics = mediapipe.solutions.holistic

holistic = mp_holistics.Holistic(static_image_mode=False)


holistic_results = []


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

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(bgr)

    holistic_results.append(results)

# dura.flag()
#
# print("Time to load images from cashe, draw results and save.")
# for i in range(100):
#
#     print(f"{i}/100")
#
#     frame = get_frame_RGB(i)
#
#     path = f"data/video_Hol/frame_{i}.png"
#
#     cv2.imwrite(path, frame)
#
#     results


dura.flag()
dura.head()


successes = [
    results.right_hand_landmarks is not None for results in holistic_results
]

count = numpy.sum(successes)

print(successes)
print(count)


# static_image_mode=True

# EvaluateHolistics.py:44 -> EvaluateHolistics.py:54 = 6971.7ms.
# EvaluateHolistics.py:54 -> EvaluateHolistics.py:85 = 23431.2ms.

# 23431.2ms. - 6971.7ms. = 16459.5



# 16459.5 ms for for 100 images

# 164.595 ms per frame

# 6.0755186974 FPS

# 73% Success

# static_image_mode=False

# EvaluateHolistics.py:44 -> EvaluateHolistics.py:54 = 7255.64ms.
# EvaluateHolistics.py:54 -> EvaluateHolistics.py:85 = 19247.04ms.

# 19247.04 - 7255.64 = 11991.4

# 11991.4 for 100
# 119.914 for 1

# 8.3393098387 FPS

# 78