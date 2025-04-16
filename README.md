## Testing

- Finding optimal resolution and FOV

When initialising the camera, the resolution and format are chosen.

If too low of a resolution is chosen, the image will be cropped. If too high of
a resolution is chosen, the framerate will drop.

It may be worth plotting both these values out, which should take a sort of
bodeplot looking shape. Then in the middle a value

The full resolution (4608 × 2592) requires more than 33ms(30fps) to process on
the camera modules.

- Finding optimal Camera spacing

- Finding optimal disparity settings

# Threading

Possessing two feeds in a single thread runs at roughly 22fps (11 Per frame).
Possessing two feeds in a separate thread runs at roughly 30fps (15 Per frame).

# Direction

Using
[body relative direction](https://en.wikipedia.org/wiki/Body_relative_direction)
