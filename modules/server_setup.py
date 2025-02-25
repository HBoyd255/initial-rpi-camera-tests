import queue
import socket
import struct
import sys
import threading
import time
import cv2


frame_queue = queue.Queue(maxsize=1)

IP = "144.32.70.219"
PORT = 1234


def _compress_and_send(client_socket, frame):

    params = [cv2.IMWRITE_JPEG_QUALITY, 50]
    data = cv2.imencode(".jpg", frame, params)[1].tobytes()

    # Print the size of the serialized frame in bytes
    x, y, z = frame.shape
    uncompressed_size = x * y * z
    compressed_size = sys.getsizeof(data)

    

    print(f"Sending UCMP= {uncompressed_size} bytes", end=", ")
    print(f"CMP= {compressed_size} bytes", end=", ")
    print(f"Ratio= {uncompressed_size/compressed_size:.2f}", end=", ")

    size = len(data)
    client_socket.sendall(struct.pack(">L", size))

    client_socket.sendall(data)


def _server_connection():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((IP, PORT))
    server_socket.listen(1)

    print(f"Server started on {IP}:{PORT}")

    while True:
        try:
            print("Waiting for connection...")
            client_socket, address = server_socket.accept()
            print(f"Connection from {address} established")
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
            continue

        while client_socket:
            try:
                frame = frame_queue.get()
                _compress_and_send(client_socket, frame)
            except Exception as e:
                print(f"Error: {e}")
                break


def start_server():
    server_thread = threading.Thread(target=_server_connection, daemon=True)
    server_thread.start()


def send_to_server(frame):
    if not frame_queue.full():
        frame_queue.put(frame)
