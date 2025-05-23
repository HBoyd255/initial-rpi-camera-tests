import platform
import cv2


if platform.system() == "Linux":

    import picamera2  # type: ignore
    from libcamera import Transform  # type: ignore

    class Eye:

        def __init__(self, side="left"):

            side = side.casefold()

            if side == "left" or side == "l":
                index = 1
            elif side == "right" or side == "r":
                index = 0
            else:
                # TODO Add error case.
                return

            self._cam = picamera2.Picamera2(camera_num=index)

            # For whatever reason, in this parameter RGB and BGR are flipped.
            # This module actually returns a BGR image.
            config = self._cam.create_video_configuration(
                main={"size": (2304, 1296), "format": "RGB888"}
            )
            config["transform"] = Transform(hflip=1, vflip=1)

            # TODO Do tests into best base resolution.

            self._cam.configure(config)
            self._cam.start()

        def array(self, format="bgr", res="low"):
            image = self._cam.capture_array()

            # TODO Do tests into best scaled resolution.
            if res.casefold() == "low":
                image = image[::4, ::4]

            elif res.casefold() == "mid":
                image = image[::2, ::2]

            if format.casefold() == "rgb":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image

        def jpeg(self, quality=50, res="low"):

            frame = self.array(res=res)

            params = [cv2.IMWRITE_JPEG_QUALITY, quality]

            jpeg = cv2.imencode(".jpg", frame, params)[1].tobytes()

            return jpeg

elif platform.system() == "Windows":

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

            self._cam = cv2.VideoCapture(index)

            # Set to full resolution.
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

        def array(self, format="bgr", res="full"):
            _, image = self._cam.read()

            # TODO Do tests into best scaled resolution.
            if res.casefold() == "low":
                image = image[::4, ::4]

            elif res.casefold() == "mid":
                image = image[::2, ::2]

            if format.casefold() == "rgb":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image

        def jpeg(self, quality=50, res="full"):

            frame = self.array(res=res)

            params = [cv2.IMWRITE_JPEG_QUALITY, quality]

            jpeg = cv2.imencode(".jpg", frame, params)[1].tobytes()

            return jpeg
