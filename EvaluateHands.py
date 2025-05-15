import time
import cv2

import mediapipe
import numpy
from modules.eye import Eye

from modules.zoomLive import Zoom

mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands(static_image_mode=False)

baseline_a = []
baseline_b = []
total_times = []


def get_frame(i):
    path = f"data/video/frame_{i}.png"

    frame = cv2.imread(path)

    return frame


print("Time to load images from memory and convert to RGB again")
for i in range(100):

    print(f"{i}/100")

    start = time.time()

    frame = get_frame(i)

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    end = time.time()
    duration = end - start

    baseline_a.append(duration)


print("Time to load images from memory and convert to RGB again")
for i in range(100):

    print(f"{i}/100")

    start = time.time()

    frame = get_frame(i)

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    end = time.time()
    duration = end - start

    baseline_b.append(duration)


print("Time to load images from memory, and prossess.")
for i in range(100):

    print(f"{i}/100")

    start = time.time()

    frame = get_frame(i)

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(bgr)

    end = time.time()
    duration = end - start

    total_times.append(duration)


baseline_b = numpy.array(baseline_b)
total_times = numpy.array(total_times)

proc_times = total_times - baseline_b

numpy.savetxt("1_HAN_C.txt", proc_times)





baseline_a = []
baseline_b = []
total_times = []

mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)


print("Time to load images from memory and convert to RGB again")
for i in range(100):

    print(f"{i}/100")

    start = time.time()

    frame = get_frame(i)

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    end = time.time()
    duration = end - start

    baseline_a.append(duration)


print("Time to load images from memory and convert to RGB again")
for i in range(100):

    print(f"{i}/100")

    start = time.time()

    frame = get_frame(i)

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    end = time.time()
    duration = end - start

    baseline_b.append(duration)


print("Time to load images from memory, and prossess.")
for i in range(100):

    print(f"{i}/100")

    start = time.time()

    frame = get_frame(i)

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(bgr)

    end = time.time()
    duration = end - start

    total_times.append(duration)


baseline_b = numpy.array(baseline_b)
total_times = numpy.array(total_times)

proc_times = total_times - baseline_b

numpy.savetxt("1_HAN_S.txt", proc_times)