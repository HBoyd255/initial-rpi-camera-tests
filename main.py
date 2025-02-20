import cv2
import mediapipe
import numpy
import platform

HEADLESS_MODE = False

hands = mediapipe.solutions.hands




cap = cv2.VideoCapture(0)
def get_rgb_frame() -> numpy.ndarray:

    system = platform.system()

    print(f"System is "{system})

    success, frame = cap.read()

    if not success:
        pass
        # TODO do something

    return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def main() -> int:

    while True:
        frame = get_rgb_frame()

        if not HEADLESS_MODE:
            cv2.imshow("Frame", frame)

            cv2.waitKey(1)


if __name__ == "__main__":
    main()
