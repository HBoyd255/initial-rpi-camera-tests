import cv2
from flask import Flask, Response
from modules.vision import Vision

app = Flask(__name__)
eyes = Vision()


def generate(mode):

    if mode.casefold() == "single":
        while True:

            jpeg = eyes.left.jpeg()

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    if mode.casefold() == "both":
        while True:

            frame = eyes.joined_array()
            params = [cv2.IMWRITE_JPEG_QUALITY, 50]
            jpeg = cv2.imencode(".jpg", frame, params)[1].tobytes()

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    if mode.casefold() == "onion":
        while True:

            frame = eyes.onion_array()
            params = [cv2.IMWRITE_JPEG_QUALITY, 50]
            jpeg = cv2.imencode(".jpg", frame, params)[1].tobytes()

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    if mode.casefold() == "redblue":
        while True:

            frame = eyes.anaglyph_array()
            params = [cv2.IMWRITE_JPEG_QUALITY, 90]
            jpeg = cv2.imencode(".jpg", frame, params)[1].tobytes()

            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


@app.route("/video")
def video():
    return Response(
        generate("single"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "max-age=0, no-store, no-cache, must-revalidate"
        },
    )


@app.route("/both")
def both():
    return Response(
        generate("both"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "max-age=0, no-store, no-cache, must-revalidate"
        },
    )


@app.route("/onion")
def onion():
    return Response(
        generate("onion"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "max-age=0, no-store, no-cache, must-revalidate"
        },
    )


@app.route("/redblue")
def redblue():
    return Response(
        generate("redblue"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "max-age=0, no-store, no-cache, must-revalidate"
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1985, debug=False)
