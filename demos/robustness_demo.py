import cv2
import numpy
from modules.eye import Eye

from modules.fps import FPS
from modules.video import Video
from modules.zoom import Zoom


eye = Eye()
vid = Video(canvas_framing=(2, 2))

hand_finder = Zoom()
hand_finder_en = Zoom()

fps = FPS()

while True:

    fps.tick()

    full_fame = eye.array(res="full")

    left_hand = hand_finder.get_hand(full_fame, simple=True)
    right_hand = hand_finder_en.get_hand(full_fame, simple=False)

    small_frame = full_fame[::2, ::2]

    left_feed = numpy.copy(small_frame)
    right_feed = numpy.copy(small_frame)

    right_feed = hand_finder_en.draw_zoom_outline(right_feed)
    right_feed = hand_finder_en.draw_last_body(right_feed)

    left_feed = left_hand.draw(left_feed)
    right_feed = right_hand.draw(right_feed)

    vid.show("Off the Shelf MediaPipe", left_feed, show_resolution=False)
    vid.show("Enhanced System", right_feed, show_resolution=False)

    if right_hand._seen_from_zoom or right_hand._seen == False:

        zoom_area = hand_finder_en._get_zoom_frame(full_fame)

        zoom_area = numpy.repeat(zoom_area, 2, axis=0)
        zoom_area = numpy.repeat(zoom_area, 2, axis=1)

        vid.show("Zoomed In Area", zoom_area, show_resolution=False)

    canv = numpy.zeros_like(left_feed)

    cv2.putText(
        canv,
        f"FPS = {fps.get_fps():.4}",
        (30, 100),
        cv2.FONT_HERSHEY_TRIPLEX,
        2,
        (0, 0, 255),
        2,
    )
    
    vid.show("", canv, show_resolution=False)
