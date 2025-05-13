import numpy


class Angle:

    def __init__(self, radians):
        self._radians = self._normalise(radians)

    @classmethod
    def from_deg(cls, degrees):
        return cls(numpy.deg2rad(degrees))

    def _normalise(self, value):

        return ((value + numpy.pi) % (2 * numpy.pi)) - numpy.pi

    def __add__(self, other):
        if isinstance(other, Angle):
            return Angle(self._radians + other._radians)

        else:
            return Angle(self._radians + other)

    def __sub__(self, other):
        if isinstance(other, Angle):
            return Angle(self._radians - other._radians)
        else:
            return Angle(self._radians - other)

    def __str__(self):
        return f"Angle:{self.get_degrees():.4}°"

    def get_degrees(self):
        return numpy.rad2deg(self._radians)

    def get_radians(self):
        return self._radians
