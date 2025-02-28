import queue
import socket
import struct
import threading
import time
import cv2
import numpy as np
from modules.file_utils import text_file_to_string
import keyboard

IP = text_file_to_string("secrets/IP.txt")
PORT = 1234


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP, PORT))

frame_queue = queue.Queue(maxsize=1)


def send_motor_commands(
    fl_speed: int,
    fr_speed: int,
    bl_speed: int,
    br_speed: int,
    x: int = 0,
    y: int = 0,
):

    # Pack the data in 6 bytes
    packed_data = struct.pack(
        ">4b2B", fl_speed, fr_speed, bl_speed, br_speed, x, y
    )

    # Create a value to store the last sent data, this prevents spamming the
    # server with the same data.
    if not hasattr(send_motor_commands, "timed_out_value"):
        send_motor_commands.timed_out_value = None

    # Store the time at which the data was last sent.
    if not hasattr(send_motor_commands, "last_sent_time"):
        send_motor_commands.last_sent_time = 0

    # Calculate the time since the last data was sent.
    time_since_last_sent = time.time() - send_motor_commands.last_sent_time

    # If it has been more that 100ms since the last data was sent, allow a
    # duplicate value to be sent by setting the timed_out_value to None.
    if time_since_last_sent > 0.1:
        send_motor_commands.timed_out_value = None

    # If the data is the same as the last sent data, don't send it.
    if packed_data != send_motor_commands.timed_out_value:
        client_socket.sendall(packed_data)

        print(f"Sent: {fl_speed}, {fr_speed}, {bl_speed}, {br_speed}, {x}, {y}")

        send_motor_commands.timed_out_value = packed_data
        send_motor_commands.last_sent_time = time.time()


def incoming_data():

    while True:

        data = client_socket.recv(4)
        size = int.from_bytes(data, byteorder="big")

        jpg_data = b""

        while len(jpg_data) < size:
            jpg_data += client_socket.recv(size - len(jpg_data))

        # Convert the byte array to a numpy array
        jpg_data = np.frombuffer(jpg_data, dtype=np.uint8)

        # Decode the Jpeg into an image
        frame = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)

        # Convert the image to BGR.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_queue.put(frame)


def normalize_speeds(speeds: np.ndarray) -> np.ndarray:

    # Find the maximum absolute value
    max_value = np.max(np.abs(speeds))

    if max_value > 100:
        speeds = speeds / max_value * 100

        speeds = speeds.astype(int)

    return speeds


incoming_thread = threading.Thread(target=incoming_data, daemon=True)
incoming_thread.start()


x = 50
y = 50


while True:

    # FL, FR, BL, BR
    speeds = np.zeros(4, dtype=int)

    if not frame_queue.empty():
        frame = frame_queue.get()

        cv2.imshow("Client Frame", frame)
        cv2.waitKey(1)

    if keyboard.is_pressed("w"):
        # All motors go forwards
        speeds += 100

    if keyboard.is_pressed("s"):
        # All motors go backwards
        speeds += -100

    if keyboard.is_pressed("a"):
        # Drive left,
        # Front left and back right go backwards
        # Front right and back left go forwards
        speeds += [-100, 100, 100, -100]

    if keyboard.is_pressed("d"):
        # Drive right,
        # Front left and back right go forwards
        # Front right and back left go backwards
        speeds += [100, -100, -100, 100]

    if keyboard.is_pressed("k"):
        # Rotate left
        # Left motors go backwards
        # Right motors go forwards
        speeds += [-100, 100, -100, 100]

    if keyboard.is_pressed("l"):
        # Rotate right
        # Left motors go forwards
        # Right motors go backwards
        speeds += [100, -100, 100, -100]

    speeds = normalize_speeds(speeds)

    send_motor_commands(*speeds, x, y)
