import cv2
import numpy

from modules.aruco import Aruco, aruco_list


LEFT_INDEX = 15
RIGHT_INDEX = 16


RED = (0, 0, 255)

DROP_THRESHOLD = 30


class ZoomAruco:

    def __init__(self):

        self.use_zoom = False

        self.resolution_full = None

        self._zoom_coords = numpy.array([1 / 8, 1 / 8])

        self._concurrent_failures = 0

        self._last_seen_tag_list = []

    def get_from_fov(self, frame_th) -> list[Aruco]:

        tag_from_full = aruco_list(frame_th)

        return tag_from_full

    def get_from_zoom(self, frame_f) -> list[Aruco]:

        off_x = int(self._zoom_coords[0] * self.resolution_full[0])
        off_y = int(self._zoom_coords[1] * self.resolution_full[1])

        bound_y = self.resolution_full[1] // 8
        bound_x = self.resolution_full[0] // 8

        zoom_frame = frame_f[
            off_y - bound_y : off_y + bound_y,
            off_x - bound_x : off_x + bound_x,
        ]

        tags_from_zoom = aruco_list(zoom_frame, seen_from_zoom=True)

        if tags_from_zoom:
            self._evaluate_zoom_use(numpy.copy(tags_from_zoom[0].landmarks))

        for tag in tags_from_zoom:

            tag.landmarks /= 4
            tag.landmarks += self._zoom_coords
            tag.landmarks -= numpy.array([1 / 8, 1 / 8])

        return tags_from_zoom

    def _recenter_from_tag(self, tag: Aruco):

        dif = self._zoom_coords - tag.get_centre()

        dif /= 2

        self._zoom_coords -= dif

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

    def get_tags(self, full_res_frame) -> list[Aruco]:

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

        if self._concurrent_failures > DROP_THRESHOLD:

            cf = self._concurrent_failures

            x_o = cf % 7
            y_o = cf // 7 % 7

            # print(f"cf:{cf} -> [{x_o},{y_o}]")

            self._zoom_coords = numpy.array([((x_o + 1) / 8), ((y_o + 1) / 8)])

            tags = self.get_from_zoom(full_res_frame)

            # print(tag)

            if tags:
                self._concurrent_failures = 0
                self._last_seen_tag_list = tags
                return tags

            self.use_zoom = False

        if self.use_zoom is False:
            tags = self.get_from_fov(full_fov_thumb)

            if tags:
                self._recenter_from_tag(tags[0])
                self._concurrent_failures = 0
                self._last_seen_tag_list = tags
                return tags

            self._concurrent_failures += 1
            self.use_zoom = True

            if self._concurrent_failures < DROP_THRESHOLD:
                return self._last_seen_tag_list

            return tags

        else:
            tags = self.get_from_zoom(full_res_frame)

            if tags:
                self._recenter_from_tag(tags[0])

                self._concurrent_failures = 0
                self._last_seen_tag_list = tags
                return tags

        self._concurrent_failures += 1

        # If no Aruco tags where found, return a blank tag.

        if self._concurrent_failures <= DROP_THRESHOLD:
            return self._last_seen_tag_list

        return []

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
