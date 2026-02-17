# laser_interface.py
import gpiod
import time

class LaserInterface:
    MIN_FIRE_TIME = 0.25 # seconds

    def __init__(self, chip="gpiochip0", line=17):
        self.chip = gpiod.Chip(chip)
        self.line = self.chip.get_line(line)
        self.line.request(consumer="laser", type=gpiod.LINE_REQ_DIR_OUT)
        self.ready = True

    def fire(self, aim_point):
        if not self.ready:
            return
        
        self.ready = False
        self.line.set_value(1)
        time.sleep(self.MIN_FIRE_TIME)
        self.line.set_value(0)
        self.ready = True



    def send(self, x, y):
        """
        Prints coordinates
        """
        # TODO: GPIO
        print(f"[LASER] firing at ({x:.1f}, {y:.1f})")

    