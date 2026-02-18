# gpio pulse test

import gpiod
import time

CHIP = "gpiochip0"
LINE = 4   # Pin 7, uses GPIO4 line (function #1 listed in gpio pinout)

chip = gpiod.Chip(CHIP)
line = chip.get_line(LINE)

line.request(consumer="laser_test",
             type=gpiod.LINE_REQ_DIR_OUT)

try:
    while True:
        print("Laser ON")
        line.set_value(1)
        time.sleep(0.5)

        print("Laser OFF")
        line.set_value(0)
        time.sleep(0.5)

except KeyboardInterrupt:
    pass
finally:
    line.release()
