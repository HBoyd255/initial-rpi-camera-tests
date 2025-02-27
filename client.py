import socket
import cv2
import numpy
from modules.file_utils import text_file_to_string

IP = text_file_to_string("secrets/IP.txt")
PORT = 1234

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP, PORT))

while True:
    data = client_socket.recv(4)
    size = int.from_bytes(data, byteorder="big")

    jpg_data = b""

    while len(jpg_data) < size:
        jpg_data += client_socket.recv(size - len(jpg_data))

    # Convert the byte array to a numpy array
    jpg_data = numpy.frombuffer(jpg_data, dtype=numpy.uint8)

    # Decode the Jpeg into an image
    frame = cv2.imdecode(jpg_data, cv2.IMREAD_COLOR)

    # Convert the image to BGR.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Frame", frame)

    cv2.waitKey(1)
