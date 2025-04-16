class DistanceCalculator:

    def __init__(self, baseline_m, focal_length_px, frame_width_px):
        self._baseline_m = baseline_m
        self._focal_length_px = focal_length_px
        self._frame_width_px = frame_width_px

        self._distance_numerator = (
            self._baseline_m * self._focal_length_px / self._frame_width_px
        )

    def get_disparities(self, left_eye_hand, right_eye_hand):

        disparities = (
            left_eye_hand.landmarks[:, 0] - right_eye_hand.landmarks[:, 0]
        )

        return disparities

    def get_distances(self, left_eye_hand, right_eye_hand):

        disparities = self.get_disparities(left_eye_hand, right_eye_hand)

        distances = self._distance_numerator / disparities
        
        return distances
