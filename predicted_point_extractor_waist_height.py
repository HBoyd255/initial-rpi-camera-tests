import numpy
from modules.video import Video
from modules.zoom import Zoom

from modules.data_set_control_waist_height import DataSetControlWaistHeight
from modules.localiser import Localiser

localiser = Localiser()


dscwh = DataSetControlWaistHeight()


CONTINUOUS = False
ZOOM_STATE = "low_res"

USE_SIMPLE = ZOOM_STATE in ("low_res", "full_res")
USE_FULL = ZOOM_STATE in ("full_res")


angles = dscwh.get_angles()
dists = dscwh.get_distances()

vid = Video(local=True)

for angle in angles:

    print(f"Running Trial for Angle:{angle}")

    # Reset the finders
    left_hand_finder = Zoom(continuous=CONTINUOUS)
    right_hand_finder = Zoom(continuous=CONTINUOUS)

    left_hand_finder._zoom_coords = numpy.array([1 / 2, 1 / 8])
    right_hand_finder._zoom_coords = numpy.array([1 / 2, 1 / 8])

    if CONTINUOUS:
        for i in range(3):
            left_frame, right_frame = dscwh.get_frames(dists[0], angle)

            left_hand = left_hand_finder.get_hand(
                left_frame, simple=USE_SIMPLE, use_full=USE_FULL
            )
            right_hand = right_hand_finder.get_hand(
                right_frame, simple=USE_SIMPLE, use_full=USE_FULL
            )

            left_frame = left_hand_finder.draw_zoom_outline(left_frame)
            right_frame = right_hand_finder.draw_zoom_outline(right_frame)

            left_frame = left_hand.draw(left_frame)
            right_frame = right_hand.draw(right_frame)

            both_new = numpy.hstack((left_frame, right_frame))

    for dist in dists:

        left_frame, right_frame = dscwh.get_frames(dist, angle)

        left_hand = left_hand_finder.get_hand(
            left_frame, simple=USE_SIMPLE, use_full=USE_FULL
        )
        right_hand = right_hand_finder.get_hand(
            right_frame, simple=USE_SIMPLE, use_full=USE_FULL
        )

        predicted_points = localiser.get_coords(left_hand, right_hand)

        dscwh.save_predicted(
            dist,
            angle,
            predicted_points,
            continuous=CONTINUOUS,
            zoom_state=ZOOM_STATE,
        )

        left_frame = left_hand.draw(left_frame)
        right_frame = right_hand.draw(right_frame)

        left_frame = left_frame[::4, ::4]
        right_frame = right_frame[::4, ::4]

        left_frame = left_hand_finder.draw_zoom_outline(left_frame)
        right_frame = right_hand_finder.draw_zoom_outline(right_frame)

        both_new = numpy.hstack((left_frame, right_frame))

        vid.show("both_new", both_new)

