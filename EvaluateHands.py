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


mp_hands = mediapipe.solutions.hands

hands = mp_hands.Hands(static_image_mode=True)


hands_results = []


def get_frame(i):
    path = f"data/video2/frame_{i}.png"

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

    results = hands.process(bgr)

    hands_results.append(results)

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
    results.multi_hand_landmarks is not None for results in hands_results
]

count = numpy.sum(successes)

print(successes)
print(count)



# on bad data set 0%
# static_image_mode=False)
# EvaluateHands.py:42 -> EvaluateHands.py:52 = 7253.62ms.
# EvaluateHands.py:52 -> EvaluateHands.py:82 = 12685.14ms.

# 12685.14 - 7253.62 = 5431.52

# 5431.52 per 100 images
# 54.3152 per frame

# 18.4110525231 FPS



# static_image_mode=True)
# EvaluateHands.py:42 -> EvaluateHands.py:52 = 7162.8ms.
# EvaluateHands.py:52 -> EvaluateHands.py:82 = 12588.83ms.

# 12588.83 - 7162.8 = 5426.03

# 5426.03 per 100 images
# 54.2603 per frame

# 18.4296806321 FPS




# on bad data set 100%
# static_image_mode=False)

# EvaluateHands.py:42 -> EvaluateHands.py:52 = 7240.78ms.
# EvaluateHands.py:52 -> EvaluateHands.py:82 = 15643.72ms.

# 15643.72 - 7240.78 = 8402.94

# 8402.94 per 100 images
# 84.0294 per frame

# 11.9005966959 FPS



# static_image_mode=True)
# EvaluateHands.py:42 -> EvaluateHands.py:52 = 7121.58ms.
# EvaluateHands.py:52 -> EvaluateHands.py:82 = 15205.59ms.

# 15205.59 - 7121.58 = 8084.01

# 8084.01 per 100 images
# 80.8401 per frame

# 12.3700985031 FPS

