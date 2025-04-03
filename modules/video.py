import os
from queue import Queue
from threading import Thread
import cv2


frame_queue = Queue(maxsize=1)


class Video:

    def __init__(self):

        self._display_available = "DISPLAY" in os.environ

    def show(self, name, frame):

        global frame_queue

        try:
            frame_queue.put_nowait(frame)
        except:
            # Replace old frame with new one
            frame_queue.get_nowait()
            frame_queue.put_nowait(frame)

        if not self._display_available:
            return

        cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF

        # If esc is pressed, exit the program
        if key == 27:
            exit()


import cv2
from flask import Flask, Response

app = Flask(__name__)


def generate():

    while True:

        try:
            frame = frame_queue.get_nowait()
            params = [cv2.IMWRITE_JPEG_QUALITY, 50]
            jpeg = cv2.imencode(".jpg", frame, params)[1].tobytes()

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        except:
            continue


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "max-age=0, no-store, no-cache, must-revalidate"
        },
    )


def start_server():
    app.run(host="0.0.0.0", port=1985, debug=False)


server_thread = Thread(target=start_server)
server_thread.start()
