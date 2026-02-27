# laser_interface.py
import serial #TODO replace gpiod, check that this is viable
import time

class LaserInterface:
    MIN_FIRE_TIME = 0.25 # seconds

    def __init__(self, port="/dev/serial/by-id/usb-MicroPython_Board_in_FS_mode_e6641cb2cf162f27-if00", baud=115200): #port is microcontroller port
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


    