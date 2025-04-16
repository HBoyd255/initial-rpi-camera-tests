import cv2
import numpy

from modules.eye import Eye
from modules.video import Video

# ELEVATION_OFFSET_DEGREES is around 21 degrees


aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
detector = cv2.aruco.ArucoDetector(aruco_dictionary)


eye = Eye()
vid = Video(canvas_framing=(1, 1))

VERTICAL_FOV_DEGREES = 67


while True:

    image = eye.array(res="full")

    height = len(image)
    width = len(image[0])

    corners, ids, _ = detector.detectMarkers(image)

    if ids is None:
        continue

    corners = numpy.array(corners[0][0], dtype=int)

    top_corner = corners[0]

    y_value = top_corner[1]

    y_value_percentage = y_value / height

    y_value_percentage_from_centre = y_value_percentage - 0.5

    elevation_offset_deg = y_value_percentage_from_centre * VERTICAL_FOV_DEGREES

    print(elevation_offset_deg)

    cv2.line(image, (0, height // 2), (width, height // 2), (0, 0, 255), 3)

    cv2.line(image, (0, y_value), (width, y_value), (0, 0, 255), 3)

    cv2.putText(
        image,
        f"Centre Line",
        (100, height // 2 - 30),
        cv2.FONT_HERSHEY_TRIPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        image,
        f"Global Ground Plane",
        (100, y_value - 30),
        cv2.FONT_HERSHEY_TRIPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.line(image, (500, height // 2), (500, y_value), (0, 0, 255), 3)

    cv2.putText(
        image,
        f"Elevation Offset: {elevation_offset_deg:.4} Degrees",
        (500, (y_value + (height // 2)) // 2 - 30),
        cv2.FONT_HERSHEY_TRIPLEX,
        1,
        (0, 0, 255),
        2,
    )

    #
    vid.show("frame", image)


