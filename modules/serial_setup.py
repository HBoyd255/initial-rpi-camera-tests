import struct
import serial
import serial.tools.list_ports


def find_arduino_port():
    ports = serial.tools.list_ports.comports()

    for port in ports:
        if "Nano 33 BLE" in port.description:
            return port.device


arduino_port = find_arduino_port()
BAUD_RATE = 115200


ser = serial.Serial(arduino_port, BAUD_RATE, timeout=10)
ser.reset_input_buffer()


def send_over_serial(fl_speed, fr_speed, bl_speed, br_speed):

    x = 50
    y = 50
    # TODO Add part to send the servo angles.

    trans_code = 1
    # TODO Write the transmission codes.

    packed_data = struct.pack(
        ">1B4b2B", trans_code, fl_speed, fr_speed, bl_speed, br_speed, x, y
    )

    ser.write(packed_data)

    responce = ser.read(7)

    responce_code = responce[0]

    if responce_code != (trans_code + 1):
        print("Transmissing Issue")
