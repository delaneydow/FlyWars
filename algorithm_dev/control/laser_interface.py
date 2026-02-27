# laser_interface.py
import serial 
import time

class LaserInterface:

    def __init__(self, port="/dev/serial/by-id/usb-MicroPython_Board_in_FS_mode_e6641cb2cf162f27-if00", baud=115200): #port is microcontroller port
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(10) #allow MCU reset/system reset and calibration
        self.ready = True

    def fire(self):
        if not self.ready:
            return
        
        self.ready = False
        self.ser.write(b"FIRE\n")
        self.ser.flush()
        time.sleep(0.25)
        self.ready = True


    