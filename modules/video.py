import os
from queue import Queue
from threading import Thread
import cv2
from flask import Flask, Response
import numpy


class Video:

    # Hard coded for simplicity.
    _FRAME_SHAPE = {"w": 576, "h": 324}
    _PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 50]

    def __init__(self):

        self._display_available = "DISPLAY" in os.environ

        self._canvas = numpy.zeros(
            (self._FRAME_SHAPE["h"] * 2, self._FRAME_SHAPE["w"] * 3, 3),
            numpy.uint8,
        )

        self._frame_queue = Queue(maxsize=6)
        self._name_indexes = {}

        self._app = Flask(__name__)
        self._app.add_url_rule("/video", "video", self._video)

        self._server_thread = Thread(
            target=self._app.run,
            kwargs={"host": "0.0.0.0", "port": 1985, "debug": False},
        )
        self._server_thread.start()

    def _generate(self):

        while True:

            try:
                frame = self._frame_queue.get_nowait()
                jpeg = cv2.imencode(".jpg", frame, self._PARAMS)[1].tobytes()

                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            except:
                continue

    def _video(self):
        return Response(
            self._generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "max-age=0, no-store, no-cache, must-revalidate"
            },
        )

    def show(self, name, frame):

        # Normalise frame.
        frame = numpy.array(frame, dtype=numpy.uint8)

        # Add frame name to frame.
        cv2.putText(
            frame,
            name,
            (15, 30),
            cv2.FONT_HERSHEY_TRIPLEX,
            1,
            (0, 0, 255),
            2,
        )

        # Get the index from the frame name.
        if not name in self._name_indexes:
            self._name_indexes[name] = len(self._name_indexes)

        index = self._name_indexes[name]

        # Calculate the position on the canvas of the frame.
        offset_x = self._FRAME_SHAPE["w"] * (index % 3)
        offset_y = self._FRAME_SHAPE["h"] * (index // 3)

        self._canvas[
            offset_y : offset_y + self._FRAME_SHAPE["h"],
            offset_x : offset_x + self._FRAME_SHAPE["w"],
        ] = frame

        try:
            self._frame_queue.put_nowait(self._canvas)
        except:
            # Replace old frame with new one.
            self._frame_queue.get_nowait()
            self._frame_queue.put_nowait(self._canvas)

        if not self._display_available:
            return

        cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF

        # If esc is pressed, exit the program
        if key == 27:
            exit()
