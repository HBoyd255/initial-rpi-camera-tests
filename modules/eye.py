import cv2
import picamera2


class Eye:

    def __init__(self, side="left"):

        side = side.casefold()

        if side == "left" or side == "l":
            index = 0
        elif side == "right" or side == "r":
            index = 1
        else:
            # TODO Add error case.
            return

        self.cam = picamera2.Picamera2(camera_num=index)

        # For wahtever reason, in this parameter RGB and BGR are flipped. This
        # module actually returns a BGR image.
        config = self.cam.create_video_configuration(
            main={"size": (2304, 1296), "format": "RGB888"}
        )
        # TODO do tests into best base resolution

        self.cam.configure(config)
        self.cam.start()

    def array(self, format="bgr"):
        full_res = self.cam.capture_array()

        # TODO do tests into best scaled resolution.
        scaled_down = full_res[::4, ::4]

        if format.casefold() == "rgb":
            scaled_down = cv2.cvtColor(scaled_down, cv2.COLOR_BGR2RGB)

        return scaled_down

    def jpeg(self, quality=50):

        frame = self.array()

        params = [cv2.IMWRITE_JPEG_QUALITY, quality]

        jpeg = cv2.imencode(".jpg", frame, params)[1].tobytes()

        return jpeg
