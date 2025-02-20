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
    config = pi_cam.create_video_configuration()
    config["transform"] = Transform(hflip=MIRROR_CAMERA, vflip=1)
    pi_cam.configure(config)
    pi_cam.start()

    # TODO: add
    # FRAME_WIDTH =
    # FRAME_HEIGHT =

    def get_rgb_frame() -> numpy.ndarray:

        return pi_cam.capture_array()


def finger_bend(A, B, C, D):

    # https://mathsathome.com/angle-between-two-vectors/

    A = numpy.array(A)
    B = numpy.array(B)
    C = numpy.array(C)
    D = numpy.array(D)

    AB = B - A
    CD = D - C

    # print(BC)

    angle_AB = numpy.arctan2(*AB)
    angle_CD = numpy.arctan2(*CD)

    angle_diff = angle_CD - angle_AB  # Radians

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

    knuckle_indexes = [2, 5, 9, 13, 17]
    bend_indexes = [3, 7, 11, 15, 19]
    tip_indexes = [4, 8, 12, 16, 20]

    wrist = landmarks[0]
    knuckle = landmarks[knuckle_indexes[finger_index]]
    bend = landmarks[bend_indexes[finger_index]]
    tip = landmarks[tip_indexes[finger_index]]

    bend = finger_bend(wrist, knuckle, bend, tip)

    return bend <= bend_threshold


def process_image(frame: numpy.ndarray) -> numpy.ndarray:

    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = hand_landmarks.landmark

            coordinates = [
                (int(landmark.x * FRAME_WIDTH), int(landmark.y * FRAME_HEIGHT))
                for landmark in landmarks
            ]

            gesture_idex = 0

            #  Thumb
            for x in range(5):
                gesture_idex |= is_pointed(coordinates, x) << x

                knuckle_indexes = [2, 5, 9, 13, 17]

                cv2.circle(
                    frame,
                    coordinates[knuckle_indexes[x]],
                    10,
                    ((0, 255, 0) if ((gesture_idex >> x) & 1) else (255, 0, 0)),
                    -1,
                )

            gestures = [""] * 32
            gestures[0b00000] = "Fist"
            gestures[0b00001] = "Thumb"
            gestures[0b00010] = "Index"
            gestures[0b00011] = "German Two"
            gestures[0b00100] = "Middle"
            gestures[0b00101] = "Middle and Thumb"
            gestures[0b00110] = "Peace"
            gestures[0b00111] = "German Three"
            gestures[0b01000] = "Ring"
            gestures[0b01001] = "Ring and Thumb"
            gestures[0b01010] = "Ring and Index"
            gestures[0b01011] = "Ring, Index and Thumb"
            gestures[0b01100] = "Ring and Middle"
            gestures[0b01101] = "Ring and Middle and Thumb"
            gestures[0b01110] = "American Three"
            gestures[0b01111] = "German Four"
            gestures[0b10000] = "Pinky"
            gestures[0b10001] = "Call Me"
            gestures[0b10010] = "Rock On"
            gestures[0b10011] = "Spider-man"
            gestures[0b10100] = "Middle and Pinky"
            gestures[0b10101] = "Pinky, Middle and Thumb"
            gestures[0b10110] = "Index, Middle and Pinky"
            gestures[0b10111] = "Index, Middle, Pinky and Thumb"
            gestures[0b11000] = "Pinky and Ring"
            gestures[0b11001] = "Pinky, Ring and Thumb"
            gestures[0b11010] = "Pinky, Ring and Index"
            gestures[0b11011] = "Pinky, Ring, Index and Thumb"
            gestures[0b11100] = "Pinky, Ring and Middle"
            gestures[0b11101] = "Pinky, Ring, Middle and Thumb"
            gestures[0b11110] = "American Four"
            gestures[0b11111] = "German Five"

            cv2.putText(
                frame,
                gestures[gesture_idex],
                coordinates[0],
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            print(gesture_idex)

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

        # print(f"Processing Time: {(processing_time * 1000):.2f}MS")

        time.sleep(processing_time * DELAY_MULTIPLIER)


if __name__ == "__main__":
    main()


# Colour formats

# RGB images are the default for this project, as they are used by both the RPi
# camera and Mediapipe. However, CV2 uses BGR for both reading from the camera
# and displaying images.
