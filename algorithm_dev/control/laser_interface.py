# laser_interface.py
import serial #TODO replace gpiod, check that this is viable
import time

class LaserInterface:
    MIN_FIRE_TIME = 0.25 # seconds

    def __init__(self, port="/dev/ttyACM0", baud=115200): #port is microcontroller port
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2) #allow MCU reset
        self.ready = True

    def fire(self):
        if not self.ready:
            return
        
        self.ready = False
        self.ser.write(b"FIRE\n")
        self.ser.flush()
        self.ready = True

    def send(self, x, y):
        """
        Prints coordinates
        """
        print(f"[LASER] firing at ({x:.1f}, {y:.1f})")

    