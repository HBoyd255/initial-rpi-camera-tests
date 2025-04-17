import socket
import struct
import time
import numpy
from modules.file_utils import text_file_to_string
import keyboard

IP = text_file_to_string("secrets/IP.txt")
PORT = 1234


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP, PORT))


def send_motor_commands(
    fl_speed: int, fr_speed: int, bl_speed: int, br_speed: int
):

    # Pack the data in 6 bytes
    packed_data = struct.pack(">4b", fl_speed, fr_speed, bl_speed, br_speed)

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

        print(f"Sent: {fl_speed}, {fr_speed}, {bl_speed}, {br_speed}")

        send_motor_commands.timed_out_value = packed_data
        send_motor_commands.last_sent_time = time.time()


def normalize_speeds(speeds: numpy.ndarray) -> numpy.ndarray:

    # Find the maximum absolute value
    max_value = numpy.max(numpy.abs(speeds))

    if max_value > 100:
        speeds = speeds / max_value * 100

        speeds = speeds.astype(int)

    return speeds


MOTOR_SPEED = 100

while True:

    # FL, FR, BL, BR
    speeds = numpy.zeros(4, dtype=int)

    if keyboard.is_pressed("w"):
        # All motors go forwards
        speeds += [MOTOR_SPEED, MOTOR_SPEED, MOTOR_SPEED, MOTOR_SPEED]

    if keyboard.is_pressed("s"):
        # All motors go backwards
        speeds += [-MOTOR_SPEED, -MOTOR_SPEED, -MOTOR_SPEED, -MOTOR_SPEED]

    if keyboard.is_pressed("a") or keyboard.is_pressed("k"):
        # Rotate left
        # Left motors go backwards
        # Right motors go forwards
        speeds += [-MOTOR_SPEED, MOTOR_SPEED, -MOTOR_SPEED, MOTOR_SPEED]

    if keyboard.is_pressed("d") or keyboard.is_pressed("l"):
        # Rotate right
        # Left motors go forwards
        # Right motors go backwards
        speeds += [MOTOR_SPEED, -MOTOR_SPEED, MOTOR_SPEED, -MOTOR_SPEED]

    if keyboard.is_pressed("q"):
        # Drive left,
        # Front left and back right go backwards
        # Front right and back left go forwards
        speeds += [-MOTOR_SPEED, MOTOR_SPEED, MOTOR_SPEED, -MOTOR_SPEED]

    if keyboard.is_pressed("e"):
        # Drive right,wdd
        # Front left and back right go forwards
        # Front right and back left go backwards
        speeds += [MOTOR_SPEED, -MOTOR_SPEED, -MOTOR_SPEED, MOTOR_SPEED]

    speeds = normalize_speeds(speeds)

    send_motor_commands(*speeds)
