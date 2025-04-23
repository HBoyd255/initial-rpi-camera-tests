import numpy
from modules.aruco import aruco_list
from modules.eye import Eye


from modules.localiser import Localiser
from modules.fps import FPS
from modules.video import Video

from modules.zoomAruco import ZoomAruco


localiser = Localiser()


fps = FPS()
eye = Eye()
vid = Video()

tag_finder = ZoomAruco()


while True:

    print(fps)

    frame = eye.array(res="full")

    tags = tag_finder.get_tags(frame)

    # tags = aruco_list(frame)

    frame = numpy.array(frame[::4, ::4])

    frame = tag_finder.draw_zoom_outline(frame)

    for tag in tags:
        frame = tag.draw(frame)
        print(tag.id, end=" ")

    print()

    vid.show("frame", frame)
