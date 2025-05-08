import time
import cv2
from modules.eye import Eye


eye = Eye("left")


print(3)
time.sleep(1)
print(1)
time.sleep(1)
print(1)
time.sleep(1)

for i in range(100):

    print(f"{i}/100")

    frame = eye.array(res="full")

    path = f"data/video2/frame_{i}.png"

    cv2.imwrite(path, frame)
