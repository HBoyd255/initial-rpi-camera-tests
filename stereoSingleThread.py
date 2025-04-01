import cv2
import mediapipe
import numpy
from modules.eye import Eye
from modules.fps import FPS
from modules.hand import create_hands_list

mp_hands = mediapipe.solutions.hands
mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
hands_left = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands_right = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

left_feed = Eye("left")
right_feed = Eye("right")


def show_rgb(name: str, image: numpy.ndarray) -> None:

    # Covert to BGR for displaying
    BGR_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imshow(name, BGR_frame)

    cv2.waitKey(1)


fps_counter = FPS()


def main() -> int:

    while True:

        frame_l = left_feed.array("RGB")
        frame_r = right_feed.array("RGB")

        results_l = hands_left.process(frame_l)
        results_r = hands_right.process(frame_r)

        hands_list_l = create_hands_list(results_l)
        hands_list_r = create_hands_list(results_r)

        for hand in hands_list_l:
            hand.draw(frame_l)

        for hand in hands_list_r:
            hand.draw(frame_r)

        merged = cv2.addWeighted(frame_l, 0.5, frame_r, 0.5, 0)

        show_rgb("l", frame_l)
        show_rgb("r", frame_r)
        show_rgb("m", merged)

        fps_counter.tick()

        print(f"FPS= {fps_counter.get_fps():.4}")


if __name__ == "__main__":
    main()
