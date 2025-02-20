import cv2
import mediapipe
import numpy
import platform

HEADLESS_MODE = False

hands = mediapipe.solutions.hands


if platform.system() == "Windows":
    cap = cv2.VideoCapture(0)


    def get_rgb_frame() -> numpy.ndarray:

        success, frame = cap.read()

        if not success:
            pass
            # TODO do something

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



def main() -> int:

    while True:
        frame = get_rgb_frame()

        if not HEADLESS_MODE:

            # Covert to BGR for displaying
            BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Frame", BGR)

            cv2.waitKey(1)


if __name__ == "__main__":
    main()
