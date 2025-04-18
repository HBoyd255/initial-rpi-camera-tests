import cv2
import mediapipe
import numpy

from modules.hand import Hand
from modules.body import Body


LEFT_INDEX = 15
RIGHT_INDEX = 16


RED = (0, 0, 255)


def draw_zoom_outline(frame, offset):

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


class Zoom:

    def __init__(self, continuous=True):

        self.use_zoom = False
        self._recapture_body = False

        self.resolution_full = None

        self._zoom_coords = numpy.array([1 / 8, 1 / 8])

        self._left_is_dominant = True

        self._concurrent_failures = 0

        self._hand_mp_full = mediapipe.solutions.hands.Hands(
            static_image_mode=(not continuous),
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.7,
        )

        self._hand_mp_zoom = mediapipe.solutions.hands.Hands(
            static_image_mode=(not continuous),
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.7,
        )

        self._pose_mp = mediapipe.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
        )

    def get_from_fov(self, frame_th) -> Hand:

        full_hand_results = self._hand_mp_full.process(
            cv2.cvtColor(frame_th, cv2.COLOR_BGR2RGB)
        )

        hand_from_full = Hand(full_hand_results, seen_from_zoom=False)

        return hand_from_full

    def get_from_zoom(self, frame_f, mute=False) -> Hand:

        off_x = int(self._zoom_coords[0] * self.resolution_full[0])
        off_y = int(self._zoom_coords[1] * self.resolution_full[1])

        bound_y = self.resolution_full[1] // 8
        bound_x = self.resolution_full[0] // 8

        zoom_frame = frame_f[
            off_y - bound_y : off_y + bound_y,
            off_x - bound_x : off_x + bound_x,
        ]

        zoom_hand_results = self._hand_mp_zoom.process(
            cv2.cvtColor(zoom_frame, cv2.COLOR_BGR2RGB)
        )

        hand_from_zoom = Hand(zoom_hand_results, seen_from_zoom=True)

        if hand_from_zoom.is_seen():

            self._evaluate_zoom_use(numpy.copy(hand_from_zoom.landmarks))

            hand_from_zoom.landmarks /= 4
            hand_from_zoom.landmarks += self._zoom_coords
            hand_from_zoom.landmarks -= numpy.array([1 / 8, 1 / 8])

        return hand_from_zoom

    def _get_pose(self, frame_th) -> Body:
        pose_results = self._pose_mp.process(
            cv2.cvtColor(frame_th, cv2.COLOR_BGR2RGB)
        )
        body = Body(pose_results)

        return body

    def _recenter_from_hand(self, hand):

        dif = self._zoom_coords - hand.get_centre()

        dif /= 2

        self._zoom_coords -= dif

        self._zoom_coords = numpy.clip(self._zoom_coords, 1 / 8, 7 / 8)

    def _recenter_from_body(self, body, use_left):

        index = LEFT_INDEX if use_left else RIGHT_INDEX

        self._zoom_coords = body.landmarks[index]
        self._zoom_coords = numpy.clip(self._zoom_coords, 1 / 8, 7 / 8)

    def _evaluate_zoom_use(self, landmarks):
        y_values = landmarks[:, 1]
        x_values = landmarks[:, 0]

        max_y = numpy.max(y_values)
        min_y = numpy.min(y_values)

        max_x = numpy.max(x_values)
        min_x = numpy.min(x_values)

        max_ = numpy.array([max_x, max_y])
        min_ = numpy.array([min_x, min_y])

        range_ = max_ - min_

        max_span = numpy.max(range_)

        if max_span > 0.8:
            self.use_zoom = False

        if numpy.min(min_) < 0:
            self.use_zoom = False

        if numpy.max(max_) > 1:
            self.use_zoom = False

    def get_hand(self, full_res_frame, simple=False) -> Hand:

        # TODO move into constants

        if self.resolution_full is None:

            self.resolution_full = numpy.array(
                [
                    full_res_frame.shape[1],
                    full_res_frame.shape[0],
                ]
            )

        full_fov_thumb = numpy.array(
            full_res_frame[::4, ::4], dtype=numpy.uint8
        )

        if simple:
            return self.get_from_fov(full_fov_thumb)

        if self._concurrent_failures > 3:

            print("Recapturing Body")
            self._left_is_dominant = not self._left_is_dominant
            body = self._get_pose(full_fov_thumb)

            if body.is_seen():
                self._recapture_body = False

                self._recenter_from_body(body, use_left=self._left_is_dominant)

                hand = self.get_from_zoom(full_res_frame, mute=True)

                if hand.is_seen():
                    self._concurrent_failures = 0
                    return hand

                self.use_zoom = False

        if self.use_zoom is False:
            hand = self.get_from_fov(full_fov_thumb)

            if hand.is_seen():
                self._concurrent_failures = 0
                self._recenter_from_hand(hand)

            else:
                self._concurrent_failures += 1
                self.use_zoom = True

            return hand

        else:
            hand = self.get_from_zoom(full_res_frame)

            if hand.is_seen():
                self._concurrent_failures = 0

                self._recenter_from_hand(hand)

                return hand

        self._concurrent_failures += 1

        # If no hands where found, return a blank hand.

        return Hand(None)
