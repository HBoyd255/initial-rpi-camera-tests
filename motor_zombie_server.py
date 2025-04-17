import socket
import struct
from threading import Thread
import time
from modules.eye import Eye
from modules.file_utils import text_file_to_string

from modules.fps import FPS
from modules.video import Video
from modules.wheels import Wheels

wheel_control = Wheels()

IP = text_file_to_string("secrets/IP.txt")
PORT = 1234

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((IP, PORT))
server_socket.listen(1)

print(f"Server started on {IP}:{PORT}")


left_cam = Eye()
vid = Video(canvas_framing=(1, 1))


fps = FPS()


def stream_video():

    while True:

        frame = left_cam.array(res="mid")
        vid.show("live Feed", frame)


stream_thread = Thread(target=stream_video)
stream_thread.start()


while True:
    try:
        print("Waiting for connection...")
        client_socket, address = server_socket.accept()
        print(f"Connection from {address} established")
        while True:
            data = client_socket.recv(4)

            a = list(struct.unpack(">4b", data))

            fl, fr, bl, br = a[0], a[1], a[2], a[3]

            print(f"Received: {fl}, {fr}, {bl}, {br}")

            wheel_control.send(fl, fr, bl, br)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)
        continue
