from collections import namedtuple
import cv2
import mediapipe
from modules.eye import Eye
from modules.fps import FPS
from modules.hand import create_hands_list
from queue import Queue
from threading import Thread

mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles

FrameStruct = namedtuple("FrameStruct", ["frame", "fps"])

left_queue = Queue(maxsize=1)
right_queue = Queue(maxsize=1)


def process_frame(side: str, queue: Queue):

    eye = Eye(side)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    fps = FPS()

    while True:

        frame = eye.array("RGB")
        results = hands.process(frame)
        hands_list = create_hands_list(results)

        for hand in hands_list:
            hand.draw(frame)

        fps.tick()

        queue.put(FrameStruct(frame, fps.get_fps()))


def show():

    left_fps = 0.0
    right_fps = 0.0

    while True:

        if not left_queue.empty():
            left_frame, left_fps = left_queue.get()
            BGR_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Left", BGR_frame)
            print(f"Left FPS= {left_fps:.4} Right FPS= {right_fps:.4}")

        if not right_queue.empty():
            right_frame, right_fps = right_queue.get()
            BGR_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Right", BGR_frame)
            print(f"Left FPS= {left_fps:.4} Right FPS= {right_fps:.4}")

        cv2.waitKey(1)


if __name__ == "__main__":

    left_thread = Thread(target=process_frame, args=("left", left_queue))
    right_thread = Thread(target=process_frame, args=("right", right_queue))

    display_thread = Thread(target=show)

    left_thread.start()
    right_thread.start()
    display_thread.start()

    left_thread.join()
    right_thread.join()
    display_thread.join()
