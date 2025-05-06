from functools import lru_cache
import os
import cv2
import numpy


class DataSetControl:

    def __init__(self):
        self._angles = tuple(range(0, 360, 10))
        self._distances = tuple(range(30, 400, 10))

    def save_predicted(
        self, distance, angle, points, continuous: bool, zoom_state: str
    ):

        zoom_state = zoom_state.casefold()

        if zoom_state not in ("zoom", "low_res", "full_res"):
            print('zoom_state not "zoom", "low_res" or "full_res"')
            return

        if continuous:
            if zoom_state == "zoom":
                target_directory_name = "continuous_zoom"
            if zoom_state == "low_res":
                target_directory_name = "continuous_low_res"
            if zoom_state == "full_res":
                target_directory_name = "continuous_full_res"

        else:
            if zoom_state == "zoom":
                target_directory_name = "non_continuous_zoom"
            if zoom_state == "low_res":
                target_directory_name = "non_continuous_low_res"
            if zoom_state == "full_res":
                target_directory_name = "non_continuous_full_res"

        directory_local_path = f"data/predicted/{target_directory_name}"

        os.makedirs(directory_local_path, exist_ok=True)

        file_name = (
            f"{directory_local_path}/{distance}cm_{angle}_deg_predicted.txt"
        )

        numpy.savetxt(file_name, points, delimiter=",")

    def load_predicted(
        self, distance, angle, continuous: bool, zoom_state: str
    ):
        zoom_state = zoom_state.casefold()
        if zoom_state not in ("zoom", "low_res", "full_res"):
            print('zoom_state not "zoom", "low_res" or "full_res"')
            return

        if continuous:
            target_directory_name = f"continuous_{zoom_state}"
        else:
            target_directory_name = f"non_continuous_{zoom_state}"

        file_name = f"data/predicted/{target_directory_name}/{distance}cm_{angle}_deg_predicted.txt"

        return numpy.loadtxt(file_name, dtype=float, delimiter=",")

    @lru_cache(maxsize=512)
    def get_frames(self, distance, angle):

        left_file_name = f"data/centre_height/{distance}cm_{angle}_deg_left.png"
        left_frame = cv2.imread(left_file_name)

        right_file_name = (
            f"data/centre_height/{distance}cm_{angle}_deg_right.png"
        )
        right_frame = cv2.imread(right_file_name)

        return left_frame, right_frame

    def get_true_points(self, distance, angle):
        file = f"data/centre_height/{distance}cm_{angle}_deg_points.txt"
        return numpy.loadtxt(file, dtype=float, delimiter=",")

    def get_angles(self):
        return self._angles

    def get_distances(self):
        return self._distances
