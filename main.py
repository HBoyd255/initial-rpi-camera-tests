import argparse
import cv2
import mediapipe
import numpy
import platform
import time


parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless",
    action="store_true",
    help="Runs the program without displaying the camera feed.",
)

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

GESTURES = [""] * 32
GESTURES[0b00000] = "Fist"
GESTURES[0b00001] = "Thumb"
GESTURES[0b00010] = "Index"
GESTURES[0b00011] = "German Two"
GESTURES[0b00100] = "Middle"
GESTURES[0b00101] = "Middle and Thumb"
GESTURES[0b00110] = "Peace"
GESTURES[0b00111] = "German Three"
GESTURES[0b01000] = "Ring"
GESTURES[0b01001] = "Ring and Thumb"
GESTURES[0b01010] = "Ring and Index"
GESTURES[0b01011] = "Ring, Index and Thumb"
GESTURES[0b01100] = "Ring and Middle"
GESTURES[0b01101] = "Ring and Middle and Thumb"
GESTURES[0b01110] = "American Three"
GESTURES[0b01111] = "German Four"
GESTURES[0b10000] = "Pinky"
GESTURES[0b10001] = "Call Me"
GESTURES[0b10010] = "Rock On"
GESTURES[0b10011] = "Spider-man"
GESTURES[0b10100] = "Middle and Pinky"
GESTURES[0b10101] = "Pinky, Middle and Thumb"
GESTURES[0b10110] = "Index, Middle and Pinky"
GESTURES[0b10111] = "Index, Middle, Pinky and Thumb"
GESTURES[0b11000] = "Pinky and Ring"
GESTURES[0b11001] = "Pinky, Ring and Thumb"
GESTURES[0b11010] = "Pinky, Ring and Index"
GESTURES[0b11011] = "Pinky, Ring, Index and Thumb"
GESTURES[0b11100] = "Pinky, Ring and Middle"
GESTURES[0b11101] = "Pinky, Ring, Middle and Thumb"
GESTURES[0b11110] = "American Four"
GESTURES[0b11111] = "Halt"

if platform.system() == "Windows":

    # Temporary way to simulate the possessing delay of the pi
    # When benchmarked against each other, the RPi 4B was 6 times slower at
    # processing images than the Windows laptop I am using.
    # DELAY_MULTIPLIER = 5
    DELAY_MULTIPLIER = 0

    cap = cv2.VideoCapture(0)

    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    # For a reason unknown to me, BGRFist88 gives images in RGB format, and RGB888
    # gives BGR images. I may investigate this later, but for now it works just
    # to request to opposite format.
    config = pi_cam.create_video_configuration({"format": "BGR888"})
    config["transform"] = Transform(hflip=not MIRROR_CAMERA, vflip=1)
    pi_cam.configure(config)
    pi_cam.start()

    template_image = pi_cam.capture_array()

    FRAME_HEIGHT = len(template_image)
    FRAME_WIDTH = len(template_image[0])

    del template_image

    def get_rgb_frame() -> numpy.ndarray:

        return pi_cam.capture_array()


def finger_bend(A, B, C) -> float:

    A = numpy.array(A)
    B = numpy.array(B)
    C = numpy.array(C)

    AB = B - A
    BC = C - B

    # print(BC)

    angle_AB = numpy.arctan2(*AB)
    angle_BC = numpy.arctan2(*BC)

    angle_diff = angle_BC - angle_AB  # Radians

    # Normalize the angle to be between -pi and pi
    angle_diff = numpy.mod(angle_diff + numpy.pi, 2 * numpy.pi) - numpy.pi

    degrees = numpy.degrees(angle_diff)

    return abs(degrees)


def show_rgb(name: str, image: numpy.ndarray) -> None:

    # Covert to BGR for displaying
    BGR_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow(name, BGR_frame)

    cv2.waitKey(1)


def is_pointed(landmarks: list, finger_index: int) -> bool:

    bend_threshold = 60

    bend_indexes = [2, 6, 10, 14, 18]
    tip_indexes = [4, 8, 12, 16, 20]

    wrist = landmarks[0]
    knuckle = landmarks[bend_indexes[finger_index]]
    tip = landmarks[tip_indexes[finger_index]]

    bend = finger_bend(wrist, knuckle, tip)

    return bend <= bend_threshold


def process_image(frame: numpy.ndarray) -> [str, numpy.ndarray]:

    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = hand_landmarks.landmark

            coordinates = [
                (int(landmark.x * FRAME_WIDTH), int(landmark.y * FRAME_HEIGHT))
                for landmark in landmarks
            ]

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            gesture_index = 0

            #  Thumb
            for x in range(5):
                gesture_index |= is_pointed(coordinates, x) << x

                knuckle_indexes = [2, 5, 9, 13, 17]

                cv2.circle(
                    frame,
                    coordinates[knuckle_indexes[x]],
                    10,
                    (
                        (0, 255, 0)
                        if ((gesture_index >> x) & 1)
                        else (255, 0, 0)
                    ),
                    5,
                )

            cv2.putText(
                frame,
                GESTURES[gesture_index],
                coordinates[0],
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        # print(f"detected gesture: {GESTURES[gesture_index]}")

        return GESTURES[gesture_index], frame

    return "NONE", frame


def main() -> int:

    processing_time = 0
    processing_times = []
    average_processing_time = 0
    fps = 0
    frames_to_average = 10

    while True:

        start_time = time.time()

        frame = get_rgb_frame()

        gesture, frame = process_image(frame)

        args = parser.parse_args()

        # Check if headless mode is enabled
        if not args.headless:
            show_rgb("frame", frame)

        end_time = time.time()
        processing_time = end_time - start_time

        processing_times.append(processing_time)

        if len(processing_times) == frames_to_average:
            average_processing_time = numpy.mean(processing_times)
            fps = int(1 / processing_time)
            processing_times = []

        print(
            f"P= {(average_processing_time * 1000):.0f}MS "
            f"FPS= {fps:02d}, "
            f"Gesture= {gesture}"
        )

        time.sleep(processing_time * DELAY_MULTIPLIER)


if __name__ == "__main__":
    main()


# Colour formats

# RGB images are the default for this project, as they are used by both the RPi
# camera and Mediapipe. However, CV2 uses BGR for both reading from the camera
# and displaying images.


# Frame rate

# When plugged in, running headless, my laptop averages 28 FPS, while detecting 
# hands.


