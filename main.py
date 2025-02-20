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


def process_image(frame: numpy.ndarray) -> numpy.ndarray:

    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = hand_landmarks.landmark

            coordinates = [
                (int(landmark.x * FRAME_WIDTH), int(landmark.y * FRAME_HEIGHT))
                for landmark in landmarks
            ]
            # mp_drawing.draw_landmarks(
            #     frame,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style(),
            # )

            #  Thumb
            wrist = coordinates[0]
            knuckle = coordinates[1]
            bend = coordinates[3]
            tip = coordinates[4]

            bend = finger_bend(wrist, knuckle, bend, tip)

            thumb_bend = coordinates[2]

            # Finger is bent
            if bend > 60:
                cv2.circle(frame, thumb_bend, 10, (255, 0, 0), -1)

            # Finger is not bent
            else:
                cv2.circle(frame, thumb_bend, 10, (0, 255, 0), -1)

            # Index finger
            wrist = coordinates[0]
            knuckle = coordinates[5]
            bend = coordinates[7]
            tip = coordinates[8]

            bend = finger_bend(wrist, knuckle, bend, tip)

            # Finger is bent
            if bend > 60:
                cv2.circle(frame, knuckle, 10, (255, 0, 0), -1)

            # Finger is not bent
            else:
                cv2.circle(frame, knuckle, 10, (0, 255, 0), -1)

            # Middle finger
            wrist = coordinates[0]
            knuckle = coordinates[9]
            bend = coordinates[11]
            tip = coordinates[12]

            bend = finger_bend(wrist, knuckle, bend, tip)

            # Finger is bent
            if bend > 60:
                cv2.circle(frame, knuckle, 10, (255, 0, 0), -1)

            # Finger is not bent
            else:
                cv2.circle(frame, knuckle, 10, (0, 255, 0), -1)

            # Ring finger
            wrist = coordinates[0]
            knuckle = coordinates[13]
            bend = coordinates[15]
            tip = coordinates[16]

            bend = finger_bend(wrist, knuckle, bend, tip)

            # Finger is bent
            if bend > 60:
                cv2.circle(frame, knuckle, 10, (255, 0, 0), -1)

            # Finger is not bent
            else:
                cv2.circle(frame, knuckle, 10, (0, 255, 0), -1)

            # Pinky finger
            wrist = coordinates[0]
            knuckle = coordinates[17]
            bend = coordinates[19]
            tip = coordinates[20]

            bend = finger_bend(wrist, knuckle, bend, tip)

            # Finger is bent
            if bend > 60:
                cv2.circle(frame, knuckle, 10, (255, 0, 0), -1)

            # Finger is not bent
            else:
                cv2.circle(frame, knuckle, 10, (0, 255, 0), -1)

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
