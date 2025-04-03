import struct
import serial.tools.list_ports

TRANSMISSION_CODE = 14
BAUD_RATE = 115200
DEVICE_NAME = "Nano 33 BLE"


def find_port():
    ports = serial.tools.list_ports.comports()

    for port in ports:
        if DEVICE_NAME in port.description:
            return port.device

    return None


class Wheels:

    def __init__(self, mode="quiet"):

        self._mode = mode.casefold()

        self._serial_connection = None
        self._connect()

    def _connect(self):

        if not self._mode == "quiet":
            print(f"Attempting Connection to {DEVICE_NAME}.")

        try:
            self._port = find_port()

            if self._port is None:
                self._serial_connection = None
                raise Exception(f"{DEVICE_NAME} not found")

            self._serial_connection = serial.Serial(
                self._port,
                BAUD_RATE,
                timeout=10,
            )

            self._serial_connection.reset_input_buffer()

        except Exception as e:
            if not self._mode == "quiet":
                print("Error: " + str(e))
            return

    def send(self, fl_speed, fr_speed, bl_speed, br_speed):

        fl_speed = int(fl_speed)
        fr_speed = int(fr_speed)
        bl_speed = int(bl_speed)
        br_speed = int(br_speed)

        if not self._mode == "quiet":
            print(
                f"Attempting to send {fl_speed, fr_speed, bl_speed, br_speed}."
            )

        if self._serial_connection is None:
            try:
                if not self._mode == "quiet":
                    print("No Port Available.")
                self._connect()
            except:
                pass

        try:

            packed_data = struct.pack(
                ">1B4b",
                TRANSMISSION_CODE,
                fl_speed,
                fr_speed,
                bl_speed,
                br_speed,
            )

            self._serial_connection.write(packed_data)

            response = self._serial_connection.read(5)

            response_code = response[0]

            if response_code != (TRANSMISSION_CODE + 1):
                raise Exception(f"{TRANSMISSION_CODE} mismatch.")
            elif not self._mode == "quiet":
                print("Successfull transmission.")
        except Exception as e:
            if not self._mode == "quiet":
                print("Error: " + str(e))
            self._serial_connection = None
