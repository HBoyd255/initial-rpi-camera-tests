import time
import cv2

import numpy
from modules.eye import Eye

from modules.zoomLive import Zoom

hand_finder = Zoom(continuous=True)

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

    frame_small = numpy.array(frame[::4, ::4], dtype=numpy.uint8)

    bgr = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    end = time.time()
    duration = end - start

    baseline_a.append(duration)


print("Time to load images from memory and convert to RGB again")
for i in range(100):

    print(f"{i}/100")

    start = time.time()

    frame = get_frame(i)

    frame_small = numpy.array(frame[::4, ::4], dtype=numpy.uint8)

    bgr = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    end = time.time()
    duration = end - start

    baseline_b.append(duration)


print("Time to load images from memory, and prossess.")
for i in range(100):

    print(f"{i}/100")

    start = time.time()

    frame = get_frame(i)

    hand = hand_finder.get_hand(frame)

    end = time.time()
    duration = end - start

    total_times.append(duration)


baseline_b = numpy.array(baseline_b)
total_times = numpy.array(total_times)

proc_times = total_times - baseline_b

numpy.savetxt("1_DZ_C.txt", proc_times)





baseline_a = []
baseline_b = []
total_times = []
hand_finder = Zoom(continuous=False)

print("Time to load images from memory and convert to RGB again")
for i in range(100):

    print(f"{i}/100")

    start = time.time()

    frame = get_frame(i)

    frame_small = numpy.array(frame[::4, ::4], dtype=numpy.uint8)

    bgr = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    end = time.time()
    duration = end - start

    baseline_a.append(duration)


print("Time to load images from memory and convert to RGB again")
for i in range(100):

    print(f"{i}/100")

    start = time.time()

    frame = get_frame(i)

    frame_small = numpy.array(frame[::4, ::4], dtype=numpy.uint8)

    bgr = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    end = time.time()
    duration = end - start

    baseline_b.append(duration)


print("Time to load images from memory, and prossess.")
for i in range(100):

    print(f"{i}/100")

    start = time.time()

    frame = get_frame(i)

    hand = hand_finder.get_hand(frame)

    end = time.time()
    duration = end - start

    total_times.append(duration)


baseline_b = numpy.array(baseline_b)
total_times = numpy.array(total_times)

proc_times = total_times - baseline_b

numpy.savetxt("1_DZ_S.txt", proc_times)