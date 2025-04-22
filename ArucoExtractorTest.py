import numpy
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

    frame = eye.array(res="full")

    tag = tag_finder.get_tag(frame)

    frame = numpy.array(frame[::4, ::4])

    frame = tag_finder.draw_zoom_outline(frame)

    frame = tag.draw(frame)

    vid.show("frame", frame)
