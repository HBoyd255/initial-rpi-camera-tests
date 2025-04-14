import os
import platform
import time
import cv2
import numpy
from threading import Thread, Lock
from flask import Flask, Response


class Video:

    _PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 50]

    def __init__(self, local=False):

        self._frame_height = None
        self._frame_width = None

        has_linux_display = "DISPLAY" in os.environ
        is_windows = platform.system() == "Windows"

        self._display_available = has_linux_display or is_windows

        self._local = local

        if self._local:
            return

        self._canvas = None
        self._latest_frame = None
        self._updated_canvas = False

        self._name_indexes = {}

        self._frame_lock = Lock()

        self._app = Flask(__name__)
        self._app.add_url_rule("/video", "video", self._video)

        self._server_thread = Thread(
            target=self._app.run,
            kwargs={"host": "0.0.0.0", "port": 1985, "debug": False},
            daemon=True,
        )
        self._server_thread.start()

    def _generate(self):

        while True:

            # Add a slight delay to avoid this thread hogging the processor.
            time.sleep(0.01)

            try:

                with self._frame_lock:
                    if not self._updated_canvas:
                        continue

                    frame = self._latest_frame

                _, jpeg = cv2.imencode(".jpg", frame, self._PARAMS)

                data = (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )

                self._updated_canvas = False
                yield data

            except Exception as e:
                print(e)
                continue

    def _video(self):
        return Response(
            self._generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "max-age=0, no-store, no-cache, must-revalidate"
            },
        )

    def show(self, name, frame, second_row_text=None):

        if self._frame_height is None or self._frame_width is None:
            self._frame_width = frame.shape[1]
            self._frame_height = frame.shape[0]

        # Normalise frame.
        frame = numpy.array(frame, dtype=numpy.uint8)

        # Add frame name to frame.
        cv2.putText(
            frame,
            f"{name} ({self._frame_width} x {self._frame_height})",
            (15, 30),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 0, 255),
            2,
        )
        if second_row_text is not None:
            cv2.putText(
                frame,
                second_row_text,
                (15, 60),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if not self._local:

            if self._canvas is None:
                self._canvas = numpy.zeros(
                    (self._frame_height * 2, self._frame_width * 3, 3),
                    numpy.uint8,
                )

            # Get the index from the frame name.
            if not name in self._name_indexes:
                self._name_indexes[name] = len(self._name_indexes)

            index = self._name_indexes[name]

            # Calculate the position on the canvas of the frame.
            offset_x = self._frame_width * (index % 3)
            offset_y = self._frame_height * (index // 3)

            self._canvas[
                offset_y : offset_y + self._frame_height,
                offset_x : offset_x + self._frame_width,
            ] = frame

            with self._frame_lock:
                self._updated_canvas = True
                self._latest_frame = numpy.copy(self._canvas)

        if not self._display_available:
            return

        cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF

        # If esc is pressed, exit the program
        if key == 27:
            os._exit(0)
