import cv2
import mediapipe
import numpy
from modules.eye import Eye
from modules.fps import FPS
from modules.video import Video

fps = FPS()

from modules.hand import Hand
from modules.body import Body

hand_mp_full = mediapipe.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.7,
)

hand_mp_zoom = mediapipe.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.7,
)

pose_mp = mediapipe.solutions.pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
)


USE_LEFT_WRIST = False

if USE_LEFT_WRIST:
    wrist_index = 15
else:
    wrist_index = 16


RED = (0, 0, 255)


def cross(frame, landmark):

    height = len(frame)
    width = len(frame[0])

    pixel_coords = (int(landmark[0] * width), int(landmark[1] * height))
    cv2.line(frame, (0, pixel_coords[1]), (width, pixel_coords[1]), RED, 3)
    cv2.line(frame, (pixel_coords[0], 0), (pixel_coords[0], height), RED, 3)


def draw_zoom(frame, offset):

    height = len(frame)
    width = len(frame[0])

    pixel_coords = (int(offset[0] * width), int(offset[1] * height))

    start_x = pixel_coords[0] - (width // 8)
    end_x = pixel_coords[0] + (width // 8)

    start_y = pixel_coords[1] - (height // 8)
    end_y = pixel_coords[1] + (height // 8)

    cv2.line(frame, (start_x, start_y), (start_x, end_y), RED, 1)
    cv2.line(frame, (end_x, start_y), (end_x, end_y), RED, 1)

    cv2.line(frame, (start_x, start_y), (end_x, start_y), RED, 1)
    cv2.line(frame, (start_x, end_y), (end_x, end_y), RED, 1)


eye = Eye()
vid = Video()


hands_in_full = True

REFRAME_ZOOM = False


zoom_coords = (1 / 2, 7 / 8)

using_zoom = False

while True:

    fps.tick()

    full_res = eye.array(res="full")

    # TODO move into constants
    new_width = full_res.shape[1] // 4
    new_height = full_res.shape[0] // 4
    full_width = full_res.shape[1]
    full_height = full_res.shape[0]

    full_thumb = numpy.array(full_res[::4, ::4], dtype=numpy.uint8)

    full_thumb_annotated = numpy.copy(full_thumb)
    hand_from_zoom_frame = numpy.copy(full_thumb)

    draw_zoom(full_thumb_annotated, zoom_coords)

    if using_zoom is False:

        full_hand_results = hand_mp_full.process(
            cv2.cvtColor(full_thumb, cv2.COLOR_BGR2RGB)
        )
        hand_from_full = Hand(full_hand_results)

        # Hands Seen In Full: Yes
        if hand_from_full.is_seen():
            full_thumb_annotated = hand_from_full.draw(full_thumb_annotated)
            vid.show(
                "full",
                full_thumb_annotated,
                second_row_text=f"FPS:{fps.get_fps():.04}",
            )
            continue

    using_zoom = True

    vid.show(
        "full", full_thumb_annotated, second_row_text=f"FPS:{fps.get_fps():.04}"
    )
    # Hands Seen In Full: No

    off_x = int(zoom_coords[0] * full_width)
    off_y = int(zoom_coords[1] * full_height)

    zoom_frame = full_res[
        off_y - new_height // 2 : off_y + new_height // 2,
        off_x - new_width // 2 : off_x + new_width // 2,
    ]

    zoom_hand_results = hand_mp_zoom.process(
        cv2.cvtColor(zoom_frame, cv2.COLOR_BGR2RGB)
    )

    hand_from_zoom = Hand(zoom_hand_results)

    # Hands Seen In Full: Yes
    if hand_from_zoom.is_seen():
        zoom_frame = hand_from_zoom.draw(zoom_frame)

        hand_centre = hand_from_zoom.get_centre()

        # cross(zoom_frame, hand_centre)

        centre_error = hand_centre - (0.5, 0.5)

        centre_error /= 4

        centre_error /= 2

        zoom_coords += centre_error

        if zoom_coords[0] < 1 / 8:
            zoom_coords[0] = 1 / 8

        if zoom_coords[0] > 7 / 8:
            zoom_coords[0] = 7 / 8

        if zoom_coords[1] < 1 / 8:
            zoom_coords[1] = 7 / 8

        if zoom_coords[1] > 7 / 8:
            zoom_coords[1] = 7 / 8

        y_values = hand_from_zoom.landmarks[:, 1]
        x_values = hand_from_zoom.landmarks[:, 0]

        max_y = numpy.max(y_values)
        min_y = numpy.min(y_values)

        max_x = numpy.max(x_values)
        min_x = numpy.min(x_values)

        max_ = numpy.array([max_x, max_y])
        min_ = numpy.array([min_x, min_y])

        range_ = max_ - min_

        max_span = numpy.max(range_)

        #         cross(zoom_frame, max_)
        #         cross(zoom_frame, min_)
        #
        #         print(range_)

        if max_span > 0.8:
            using_zoom = False
            continue

        localise = numpy.copy(full_thumb)

        hand_from_zoom.landmarks /= 4
        hand_from_zoom.landmarks += zoom_coords
        hand_from_zoom.landmarks -= (1 / 8, 1 / 8)

        localise = hand_from_zoom.draw(localise, small=True)

        vid.show("Zoom Frame", zoom_frame)
        vid.show("Localise", localise)

        continue

    vid.show("Zoom Frame", zoom_frame)

    # Hands Seen In Full: No

    pose_results = pose_mp.process(cv2.cvtColor(full_thumb, cv2.COLOR_BGR2RGB))

    seen_body = Body(pose_results)

    pose_annotated = numpy.copy(full_thumb)

    if seen_body is not None:
        pose_annotated = seen_body.draw(pose_annotated)

        right_wrist = seen_body.landmarks[wrist_index]

        zoom_coords = right_wrist

        if zoom_coords[0] < 1 / 8:
            zoom_coords[0] = 1 / 8

        if zoom_coords[1] < 1 / 8:
            zoom_coords[1] = 1 / 8

        if zoom_coords[0] > 7 / 8:
            zoom_coords[0] = 7 / 8

        if zoom_coords[1] > 7 / 8:
            zoom_coords[1] = 7 / 8

    vid.show("Pose Frame", pose_annotated)
