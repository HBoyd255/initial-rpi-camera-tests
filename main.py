import argparse
import cv2
import mediapipe
import numpy
import time
from modules.eye import Eye

from modules.server_setup import start_server, send_to_server



camera = Eye()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless",
    action="store_true",
    help="Runs the program without displaying the camera feed.",
)

parser.add_argument(
    "--server",
    action="store_true",
    help="Makes the camera feed available on a local server.",
)

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




def finger_bend(A, B, C) -> float:

    A = numpy.array(A)
    B = numpy.array(B)
    C = numpy.array(C)

    AB = B - A
    BC = C - B

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


def process_image(frame: numpy.ndarray) -> tuple[str, numpy.ndarray]:

    results = hands.process(frame)

    height = len(frame)
    width = len(frame[0])

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = hand_landmarks.landmark

            coordinates = [
                (int(landmark.x * width), int(landmark.y * height))
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

    args = parser.parse_args()

    if args.server:
        start_server()

    processing_time = 0
    processing_times = []
    fps = 0
    frames_to_average = 10

    while True:

        start_time = time.time()

        frame = camera.array("RGB")

        gesture, frame = process_image(frame)

        # Check if headless mode is enabled
        if not args.headless:
            show_rgb("Server Frame", frame)

        if args.server:
            send_to_server(frame)

        end_time = time.time()
        processing_time = end_time - start_time

        processing_times.append(processing_time)

        if len(processing_times) == frames_to_average:
            fps = int(1 / numpy.mean(processing_times))
            processing_times = []

        # print(
        #     f"P= {(processing_time * 1000):.0f}MS "
        #     f"FPS= {fps:02d}, "
        #     f"Gesture= {gesture}"
        # )


if __name__ == "__main__":
    main()


# Colour formats

# RGB images are the default for this project, as they are used by both the RPi
# camera and Mediapipe. However, CV2 uses BGR for both reading from the camera
# and displaying images.


# Frame rate

# When plugged in, running headless, my laptop averages 28 FPS, while detecting
# hands.

# When running headless, the RPi 4B 8GB averages 4 FPS, while detecting hands.
# This with or without an x-server running.
