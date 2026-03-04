# laser_interface.py
import serial 
import time

class LaserInterface:

    def __init__(self, port="/dev/serial/by-id/usb-MicroPython_Board_in_FS_mode_e6641cb2cf162f27-if00", baud=115200): #port is microcontroller port
        self.ser = serial.Serial(port, baud, timeout=1)
        self.ready = False #not ready until sleep completes
        time.sleep(5) #allow MCU reset/system reset and calibration
        self.ser.reset_input_buffer() #clear any boot remnants 
        self.ready = True

    def fire(self):
        if not self.ready:
            return
        
        self.ready = False
        self.ser.write(b"FIRE\n")
        self.ser.flush()
        # read MCU acknowledgment
        try:
            resp = self.ser.readline().decode().strip()
            print(f"[LASER] MCU response: '{resp}'")
        except Exception:
            pass
        time.sleep(0.25)
        self.ready = True

    def off(self): 
        try: 
            self.ser.write(b"OFF\n")
            self.ser.flush()
        except Exception:
            pass
        self.ready = True  # re-enable after off, don't permanently block firing

    
