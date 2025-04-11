import numpy
from modules.eye import Eye
from modules.video import Video
from modules.zoom import Zoom


left_eye = Eye("left")

vid = Video()

hand_finder = Zoom(video=vid)

while True:

    full_res_frame = left_eye.array(res="full")

    hand = hand_finder.get_hand(full_res_frame)

    # Thumbnail
    thumbnail = numpy.array(full_res_frame[::4, ::4], dtype=numpy.uint8)

    # print(hand)

    if hand.is_seen():
        thumbnail = hand.draw(thumbnail)

    vid.show("Output", thumbnail)
