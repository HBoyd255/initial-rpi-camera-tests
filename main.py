import cv2
import mediapipe
import numpy
import platform
import time


HEADLESS_MODE = False

hands = mediapipe.solutions.hands


if platform.system() == "Windows":

    # Temporary way to simulate the possessing delay of the pi
    # When benchmarked against each other, the RPi 4B was 6 times slower at
    # processing images than the Windows laptop I am using.
    DELAY_MULTIPLIER = 5


    cap = cv2.VideoCapture(0)

    def get_rgb_frame() -> numpy.ndarray:

        success, bgr_frame = cap.read()

        if not success:
            pass
            # TODO do something

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        return rgb_frame

else:
    # Setup for the RPi

    DELAY_MULTIPLIER = 0

    import picamera2  # type: ignore
    from libcamera import Transform  # type: ignore

    pi_cam = picamera2.Picamera2()
    config = pi_cam.create_video_configuration()
    config["transform"] = Transform(hflip=1, vflip=1)
    pi_cam.configure(config)
    pi_cam.start()

    def get_rgb_frame() -> numpy.ndarray:

        return pi_cam.capture_array()


def show_rgb(name: str, image: numpy.ndarray) -> None:

    # Covert to BGR for displaying
    BGR_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, BGR_frame)

    cv2.waitKey(1)





def main() -> int:

    processing_time = 0

    while True:

        start_time = time.time()

        frame = get_rgb_frame()

        show_rgb("frame", frame)

        end_time = time.time()
        processing_time = end_time - start_time

        print(processing_time)

        time.sleep(processing_time * DELAY_MULTIPLIER)


if __name__ == "__main__":
    main()


# Colour formats

# RGB images are the default for this project, as they are used by both the RPi
# camera and Mediapipe. However, CV2 uses BGR for both reading from the camera
# and displaying images.
