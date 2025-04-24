import inspect
import time


class Duration:

    def __init__(self):

        self._calls = []

    def _print(self):

        print()
        print("Printing Durations:")

        flag_count = len(self._calls)

        inbetween_count = flag_count - 1

        total_start_time = self._calls[0][0]
        total_end_time = self._calls[-1][0]

        total_duration = total_end_time - total_start_time

        total_duration_ms = total_duration * 1000

        total_duration_str = str(round(total_duration_ms, 2))

        print(f"Total:{total_duration_str}ms")

        for i in range(inbetween_count):

            if i + 1 == inbetween_count:
                print("Remaining time:", end="")

            start = self._calls[i]
            end = self._calls[i + 1]

            start_time = start[0]
            end_time = end[0]

            start_line = start[1]
            end_line = end[1]

            duration = end_time - start_time

            duration_ms = duration * 1000

            duration_str = str(round(duration_ms, 2))

            name = f"{start_line} -> {end_line} = {duration_str}ms."

            print(name)

    def head(self):

        frame = inspect.stack()[1]
        filepath = frame.filename
        line_number = frame.lineno
        filename = filepath.split("/")[-1]
        head_line = f"{filename}:{line_number}"

        self._calls.append((time.time(), head_line))

        self._print()

        self._calls = []

        self._calls.append((time.time(), head_line))

    def flag(self):

        frame = inspect.stack()[1]
        filepath = frame.filename
        line_number = frame.lineno
        filename = filepath.split("/")[-1]
        flag_line = f"{filename}:{line_number}"

        self._calls.append((time.time(), flag_line))
