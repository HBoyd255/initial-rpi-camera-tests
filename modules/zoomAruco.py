import cv2
import mediapipe
import numpy

from modules.aruco import Aruco
from modules.body import Body


LEFT_INDEX = 15
RIGHT_INDEX = 16


RED = (0, 0, 255)


class ZoomAruco:

    def __init__(self, continuous=True):

        self.use_zoom = False
        self._recapture_body = False

        self.resolution_full = None

        self._zoom_coords = numpy.array([1 / 8, 1 / 8])

        self._left_is_dominant = True

        self._concurrent_failures = 0

        self._pose_mp = mediapipe.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
        )

        self._last_seen_tag = Aruco(None)

    def get_from_fov(self, frame_th) -> Aruco:

        tag_from_full = Aruco(frame_th, seen_from_zoom=False)

        return tag_from_full

    def get_from_zoom(self, frame_f) -> Aruco:

        off_x = int(self._zoom_coords[0] * self.resolution_full[0])
        off_y = int(self._zoom_coords[1] * self.resolution_full[1])

        bound_y = self.resolution_full[1] // 8
        bound_x = self.resolution_full[0] // 8

        zoom_frame = frame_f[
            off_y - bound_y : off_y + bound_y,
            off_x - bound_x : off_x + bound_x,
        ]

        tag_from_zoom = Aruco(zoom_frame, seen_from_zoom=True)

        if tag_from_zoom.is_seen():

            self._evaluate_zoom_use(numpy.copy(tag_from_zoom.landmarks))

            tag_from_zoom.landmarks /= 4
            tag_from_zoom.landmarks += self._zoom_coords
            tag_from_zoom.landmarks -= numpy.array([1 / 8, 1 / 8])

        return tag_from_zoom

    def _get_pose(self, frame_th) -> Body:
        pose_results = self._pose_mp.process(
            cv2.cvtColor(frame_th, cv2.COLOR_BGR2RGB)
        )
        body = Body(pose_results)

        return body

    def _recenter_from_tag(self, tag: Aruco):

        dif = self._zoom_coords - tag.get_centre()

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

    def get_tag(self, full_res_frame, simple=False) -> Aruco:

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

                tag = self.get_from_zoom(full_res_frame)

                if tag.is_seen():
                    self._concurrent_failures = 0
                    self._last_seen_tag = tag
                    return tag

                self.use_zoom = False

        if self.use_zoom is False:
            tag = self.get_from_fov(full_fov_thumb)

            if tag.is_seen():
                self._recenter_from_tag(tag)
                self._concurrent_failures = 0
                self._last_seen_tag = tag
                return tag

            self._concurrent_failures += 1
            self.use_zoom = True

            if self._concurrent_failures == 1:
                return self._last_seen_tag

            return tag

        else:
            tag = self.get_from_zoom(full_res_frame)

            if tag.is_seen():
                self._recenter_from_tag(tag)

                self._concurrent_failures = 0
                self._last_seen_tag = tag
                return tag

        self._concurrent_failures += 1

        # If no Aruco tags where found, return a blank tag.

        if self._concurrent_failures == 1:
            return self._last_seen_tag

        return Aruco(None)

    def draw_zoom_outline(self, frame):

        drawing_frame = numpy.copy(frame)

        height = len(drawing_frame)
        width = len(drawing_frame[0])

        pixel_coords = (
            int(self._zoom_coords[0] * width),
            int(self._zoom_coords[1] * height),
        )

        start_x = pixel_coords[0] - (width // 8)
        end_x = pixel_coords[0] + (width // 8)

        start_y = pixel_coords[1] - (height // 8)
        end_y = pixel_coords[1] + (height // 8)

        cv2.line(drawing_frame, (start_x, start_y), (start_x, end_y), RED, 1)
        cv2.line(drawing_frame, (end_x, start_y), (end_x, end_y), RED, 1)

        cv2.line(drawing_frame, (start_x, start_y), (end_x, start_y), RED, 1)
        cv2.line(drawing_frame, (start_x, end_y), (end_x, end_y), RED, 1)

        return drawing_frame
