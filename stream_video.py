from flask import Flask, Response
from modules.eye import Eye

app = Flask(__name__)
cam = Eye("Left")


def generate():
    while True:

        jpeg = cam.jpeg()

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "max-age=0, no-store, no-cache, must-revalidate"
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1985, debug=False)