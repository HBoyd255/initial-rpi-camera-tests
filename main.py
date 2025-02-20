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

else:

    import picamera2
    from libcamera import Transform

    pi_cam = picamera2.Picamera2()
    config = pi_cam.create_video_configuration()
    config["transform"] = Transform(hflip = 1, vflip = 1)
    pi_cam.configure(config)
    pi_cam.start()

    def get_rgb_frame() -> numpy.ndarray:

        return pi_cam.capture_array()


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
