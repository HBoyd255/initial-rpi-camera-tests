import cv2
import mediapipe
import numpy
import platform
import time


HEADLESS_MODE = False
MIRROR_CAMERA = True


mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


if platform.system() == "Windows":

    # Temporary way to simulate the possessing delay of the pi
    # When benchmarked against each other, the RPi 4B was 6 times slower at
    # processing images than the Windows laptop I am using.
    DELAY_MULTIPLIER = 5
    # DELAY_MULTIPLIER = 0

    cap = cv2.VideoCapture(0)

    def get_rgb_frame() -> numpy.ndarray:

        success, bgr_frame = cap.read()

        if not success:
            pass
            # TODO do something

        # Flip the frame horizontally for a mirror effect
        if MIRROR_CAMERA:
            bgr_frame = cv2.flip(bgr_frame, 1)

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        return rgb_frame

else:
    # Setup for the RPi

    DELAY_MULTIPLIER = 0

    import picamera2  # type: ignore
    from libcamera import Transform  # type: ignore

    pi_cam = picamera2.Picamera2()
    config = pi_cam.create_video_configuration()
    config["transform"] = Transform(hflip=MIRROR_CAMERA, vflip=1)
    pi_cam.configure(config)
    pi_cam.start()

    def get_rgb_frame() -> numpy.ndarray:

        return pi_cam.capture_array()


def show_rgb(name: str, image: numpy.ndarray) -> None:

    # Covert to BGR for displaying
    BGR_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, BGR_frame)

    cv2.waitKey(1)


def process_image(frame: numpy.ndarray) -> numpy.ndarray:

    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks and connections on the original frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    return frame


def main() -> int:

    processing_time = 0

    while True:

        start_time = time.time()

        frame = get_rgb_frame()

        frame = process_image(frame)

        show_rgb("frame", frame)

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"Processing Time: {(processing_time * 1000):.2f}MS")

        time.sleep(processing_time * DELAY_MULTIPLIER)


if __name__ == "__main__":
    main()


# Colour formats

# RGB images are the default for this project, as they are used by both the RPi
# camera and Mediapipe. However, CV2 uses BGR for both reading from the camera
# and displaying images.
