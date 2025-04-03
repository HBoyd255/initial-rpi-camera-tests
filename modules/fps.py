import time
import numpy


UPDATE_DELAY = 0.5


class FPS:

    def __init__(self):

        self._fps = 0.0
        self._processing_time = 0.0
        self._processing_times = []

        self._last_call_time = 0.0
        self._update_time = 0.0

        self._simple_mode = False

    def tick(self, force=False):

        if self._simple_mode and not force:
            raise Exception(
                "FPS Error: Simple mode enabled by call to __str__"
                ", unable to use .tick()"
            )

        self._processing_time = time.time() - self._last_call_time
        self._last_call_time = time.time()

        self._processing_times.append(self._processing_time)

        if (time.time() - self._update_time) > UPDATE_DELAY:
            self._fps = 1 / numpy.mean(self._processing_times)
            self._processing_times = []
            self._update_time = time.time()

    def get_fps(self):
        return self._fps

    def get_processing_time(self):
        return self._processing_time

    def __str__(self):
        self._simple_mode = True

        self.tick(force=True)

        fps_count = self.get_fps()

        return f"FPS: {fps_count:.4}"
